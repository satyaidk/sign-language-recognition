"""
STAGE 3 — Export + optimize the final model for deployment.
===========================================================
Takes checkpoints/model_final.pt and produces a self-describing, framework-free
ONNX model plus an int8-quantized variant for fast CPU inference (this box has
no GPU, so CPU latency is what matters for "live" use).

Steps:
  1. load model_final.pt, rebuild the network, load weights
  2. export to ONNX (opset 17; batch axis dynamic, sequence length fixed = L)
  3. verify: onnxruntime outputs ~= PyTorch outputs (random + a real clip)
  4. quantize_dynamic -> int8 ONNX; re-verify class agreement
  5. copy model_meta.json next to the ONNX so inference needs nothing else
  6. report file sizes + measured CPU latency

Run:
    python src/training/export_optimize.py
    python src/training/export_optimize.py --no-quant         # skip int8 variant
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from config import (CKPT_DIR, EXPORT_DIR, LANDMARKS_ALL, ensure_dirs,  # noqa: E402
                    feature_config_from_dict)
from config import ModelConfig  # noqa: E402
from model import build_model, count_params  # noqa: E402
from utils import load_json, save_json  # noqa: E402

ONNX_PATH = EXPORT_DIR / "sign_model.onnx"
ONNX_INT8 = EXPORT_DIR / "sign_model.int8.onnx"


def load_final(device="cpu"):
    ckpt_path = CKPT_DIR / "model_final.pt"
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] {ckpt_path} not found - run finetune.py first.")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    fcfg = feature_config_from_dict(ck["config"]["feature"])
    mcfg = ModelConfig(**ck["config"]["model"])
    model = build_model(ck["in_dim"], ck["n_classes"], mcfg).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck, fcfg


def _real_clip_features(fcfg, spec):
    sample = next(LANDMARKS_ALL.glob("*/*.npy"), None)
    if sample is None:
        return None
    return F.to_features(np.load(sample), fcfg, spec)[None].astype(np.float32)


def export(args):
    ensure_dirs()
    model, ck, fcfg = load_final()
    spec = F.build_spec(fcfg)
    L, D, C = ck["seq_len"], ck["in_dim"], ck["n_classes"]
    print(f"[export] model: arch={ck['config']['model']['arch']} params={count_params(model):,} "
          f"L={L} D={D} classes={C}")

    dummy = torch.randn(1, L, D)
    export_kw = dict(
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    try:
        # torch>=2.x defaults to the dynamo exporter (needs onnxscript); the
        # legacy TorchScript exporter handles GRU/Transformer fine with no extra deps.
        torch.onnx.export(model, dummy, str(ONNX_PATH), dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(ONNX_PATH), **export_kw)
    print(f"[export] wrote {ONNX_PATH}")

    # ── Verify ONNX == PyTorch ──
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def torch_logits(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    def onnx_logits(s, x):
        return s.run(["logits"], {"features": x})[0]

    tests = [np.random.randn(1, L, D).astype(np.float32),
             np.random.randn(3, L, D).astype(np.float32)]
    real = _real_clip_features(fcfg, spec)
    if real is not None:
        tests.append(real)
    max_diff = max(float(np.abs(torch_logits(x) - onnx_logits(sess, x)).max()) for x in tests)
    print(f"[export] ONNX vs PyTorch max abs logit diff = {max_diff:.2e}  "
          f"({'OK' if max_diff < 1e-3 else 'MISMATCH'})")

    # ── Dynamic int8 quantization (smaller + faster on CPU) ──
    if not args.no_quant:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(ONNX_PATH), str(ONNX_INT8), weight_type=QuantType.QInt8)
            sess8 = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])
            agree = np.mean([onnx_logits(sess, x).argmax(1)[0] == onnx_logits(sess8, x).argmax(1)[0]
                             for x in tests])
            print(f"[export] wrote {ONNX_INT8}  (int8; top-1 agreement with fp32 = {agree:.0%})")
        except Exception as exc:
            print(f"[export] quantization skipped: {exc}")

    # ── Latency (single-sample CPU) ──
    x1 = np.random.randn(1, L, D).astype(np.float32)
    for _ in range(3):
        onnx_logits(sess, x1)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        onnx_logits(sess, x1)
    lat = (time.perf_counter() - t0) / n * 1000
    print(f"[export] onnxruntime fp32 latency ~ {lat:.2f} ms/clip ({1000 / lat:.0f} clips/s)")

    # ── Self-describing meta beside the ONNX ──
    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    meta["onnx"] = ONNX_PATH.name
    meta["onnx_int8"] = ONNX_INT8.name if (ONNX_INT8.exists() and not args.no_quant) else None
    meta["onnx_vs_torch_max_diff"] = max_diff
    save_json(meta, EXPORT_DIR / "model_meta.json")
    sz = lambda p: f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "-"
    print(f"[export] sizes: fp32={sz(ONNX_PATH)}  int8={sz(ONNX_INT8)}")
    print(f"[export] meta  -> {EXPORT_DIR / 'model_meta.json'}")
    print("[export] next: python src/inference/infer_video.py --video <class>/<stem>  (file test + overlay)")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: export + optimize to ONNX.")
    p.add_argument("--no-quant", action="store_true", help="skip int8 quantization")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
