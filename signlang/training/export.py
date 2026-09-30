"""
STAGE 3 — Export + optimise the final model for deployment.
===========================================================
Turns ``checkpoints/model_final.pt`` into a framework-free, self-describing ONNX
model for fast CPU inference (live use runs on a laptop CPU).

Steps
  1. load model_final.pt and rebuild the network from its saved config
  2. export to ONNX (opset 17; dynamic batch axis; sequence length fixed = L)
  3. verify ONNX Runtime == PyTorch on random inputs (batch 1 and 3) and a real clip
  4. optional int8 dynamic quantisation — measured, not assumed: ONNX Runtime does
     not quantise GRU layers, so for the default BiGRU the int8 file is only a few
     percent smaller; it is kept only when it stays in top-1 agreement
  5. write model_meta.json next to the ONNX so inference needs nothing else
  6. report file sizes + single-clip CPU latency

The legacy TorchScript exporter (``dynamo=False``) is used on purpose: the newer
dynamo exporter needs the extra ``onnxscript`` package and has open issues with
GRU layers.  PyTorch >= 2.9 prints a deprecation warning for it; that is expected.

Run:
    python -m signlang export
    python -m signlang export --no-quant
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import torch

from signlang.config import Paths, add_path_args, feature_config_from_dict, model_config_from_dict, paths_from_args
from signlang.training import features as F
from signlang.training.model import build_model, count_params
from signlang.utils import save_json

ONNX_NAME = "sign_model.onnx"
ONNX_INT8_NAME = "sign_model.int8.onnx"
PARITY_TOL = 1e-3


def load_final(paths: Paths, device="cpu"):
    """Rebuild the final model from its self-describing checkpoint."""
    ckpt_path = paths.checkpoints / "model_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run `python -m signlang finetune` first")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    fcfg = feature_config_from_dict(ck["config"]["feature"])
    mcfg = model_config_from_dict(ck["config"]["model"])
    model = build_model(ck["in_dim"], ck["n_classes"], mcfg, max_len=ck["seq_len"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck, fcfg


def _real_clip_features(paths: Paths, fcfg, spec):
    sample = next(paths.landmarks_all.glob("*/*.npy"), None) if paths.landmarks_all.exists() else None
    if sample is None:
        return None
    return F.to_features(np.load(sample), fcfg, spec)[None].astype(np.float32)


def export_onnx(model, path, L: int, D: int) -> None:
    dummy = torch.randn(1, L, D)
    kw = dict(input_names=["features"], output_names=["logits"],
              dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
              opset_version=17, do_constant_folding=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # legacy-exporter deprecation + GRU batch notes
        try:
            torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
        except TypeError:                        # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(model, dummy, str(path), **kw)


def export(paths: Paths, quantize: bool = True) -> dict:
    import onnxruntime as ort

    paths.ensure_artifact_dirs()
    model, ck, fcfg = load_final(paths)
    spec = F.build_spec(fcfg)
    L, D, C = ck["seq_len"], ck["in_dim"], ck["n_classes"]
    onnx_path, int8_path = paths.exported / ONNX_NAME, paths.exported / ONNX_INT8_NAME
    print(f"[export] model: arch={ck['config']['model']['arch']} params={count_params(model):,} "
          f"L={L} D={D} classes={C}")

    export_onnx(model, onnx_path, L, D)
    print(f"[export] wrote {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def onnx_logits(s, x):
        return s.run(["logits"], {"features": x})[0]

    rng = np.random.default_rng(0)
    tests = [rng.standard_normal((1, L, D)).astype(np.float32),
             rng.standard_normal((3, L, D)).astype(np.float32)]
    real = _real_clip_features(paths, fcfg, spec)
    if real is not None:
        tests.append(real)
    with torch.no_grad():
        max_diff = max(float(np.abs(model(torch.from_numpy(x)).numpy() - onnx_logits(sess, x)).max())
                       for x in tests)
    ok = max_diff < PARITY_TOL
    print(f"[export] ONNX vs PyTorch max |logit diff| = {max_diff:.2e}  ({'OK' if ok else 'MISMATCH'})")
    if not ok:
        raise RuntimeError(f"ONNX export does not match PyTorch (max diff {max_diff:.2e})")

    int8_name = None
    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
            sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
            agree = float(np.mean(np.concatenate(
                [onnx_logits(sess, x).argmax(1) == onnx_logits(sess8, x).argmax(1) for x in tests])))
            print(f"[export] int8 top-1 agreement with fp32 = {agree:.0%}")
            if agree >= 0.99:
                int8_name = ONNX_INT8_NAME
            else:
                int8_path.unlink(missing_ok=True)
                print("[export] int8 model disagrees with fp32 -> discarded")
        except Exception as exc:                 # quantisation is optional
            print(f"[export] quantisation skipped: {exc}")

    x1 = tests[0]
    for _ in range(5):
        onnx_logits(sess, x1)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        onnx_logits(sess, x1)
    latency_ms = (time.perf_counter() - t0) / n * 1000
    print(f"[export] onnxruntime fp32 latency ~ {latency_ms:.2f} ms/clip")

    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    meta.update({"onnx": ONNX_NAME, "onnx_int8": int8_name, "onnx_vs_torch_max_diff": max_diff,
                 "onnx_latency_ms": round(latency_ms, 3)})
    save_json(meta, paths.exported / "model_meta.json")
    size = lambda p: f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "-"  # noqa: E731
    print(f"[export] sizes: fp32={size(onnx_path)}  int8={size(int8_path) if int8_name else '-'}")
    print("[export] next: python -m signlang video --path <clip.mp4>   (file test + overlay)")
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang export", description="Stage 3: export the final model to ONNX.")
    add_path_args(p, processed=True, artifacts=True)
    p.add_argument("--no-quant", action="store_true", help="skip the int8 variant")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        export(paths_from_args(args), quantize=not args.no_quant)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
