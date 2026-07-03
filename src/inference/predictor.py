"""
Shared inference: load the exported model and turn landmarks into a sign.
========================================================================
Used by BOTH infer_video.py (file test) and infer_live.py (webcam), so the
serving path is identical everywhere.

`SignPredictor` is self-describing: it reads `model_meta.json` (written by
export_optimize.py / finetune.py), rebuilds the EXACT FeatureConfig used in
training, and runs either the ONNX model (default, fast on CPU) or the PyTorch
checkpoint.

`video_to_clip()` reuses the project's own `extract_dataset.extract_clip` +
`build_all_vector`, so a video is converted to the same (T, 1692) raw landmark
vector the dataset was built from — no separate, drift-prone code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "training"))  # config, features, utils
import features as F  # noqa: E402
from config import (CKPT_DIR, EXPORT_DIR, PROJECT_ROOT, ModelConfig,  # noqa: E402
                    feature_config_from_dict)
from utils import load_json  # noqa: E402


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class SignPredictor:
    def __init__(self, model_dir=EXPORT_DIR, backend="onnx", int8=False):
        meta_path = Path(model_dir) / "model_meta.json"
        if not meta_path.exists():
            # fall back to the checkpoint meta if export hasn't run yet
            meta_path = CKPT_DIR / "model_meta.json"
            backend = "torch"
        self.meta = load_json(meta_path)
        self.classes = self.meta["classes"]
        self.seq_len = self.meta["seq_len"]
        self.in_dim = self.meta["in_dim"]
        self.fcfg = feature_config_from_dict(self.meta["config"]["feature"])
        self.spec = F.build_spec(self.fcfg)
        self.backend = backend

        if backend == "onnx":
            import onnxruntime as ort
            name = self.meta.get("onnx_int8") if int8 else self.meta.get("onnx")
            name = name or "sign_model.onnx"
            self.sess = ort.InferenceSession(str(Path(model_dir) / name),
                                             providers=["CPUExecutionProvider"])
        elif backend == "torch":
            import torch
            from model import build_model
            ck = torch.load(CKPT_DIR / "model_final.pt", map_location="cpu", weights_only=False)
            self.torch = torch
            self.model = build_model(ck["in_dim"], ck["n_classes"],
                                     ModelConfig(**ck["config"]["model"])).eval()
            self.model.load_state_dict(ck["state_dict"])
        else:
            raise ValueError(f"unknown backend: {backend}")

    # ── feature + logits ──
    def features(self, raw_all_vec):
        return F.to_features(raw_all_vec, self.fcfg, self.spec)

    def _logits(self, feats):
        x = feats[None].astype(np.float32)
        if self.backend == "onnx":
            return self.sess.run(["logits"], {"features": x})[0][0]
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x)).numpy()[0]

    def _probs(self, feats):
        return softmax(self._logits(feats))

    def _result(self, probs, topk):
        order = np.argsort(probs)[::-1]
        i0 = int(order[0])
        i1 = int(order[1]) if probs.shape[0] > 1 else i0
        return {"label": i0, "name": self.classes[i0], "conf": float(probs[i0]),
                "margin": float(probs[i0] - probs[i1]), "probs": probs,
                "topk": [(self.classes[int(i)], float(probs[int(i)])) for i in order[:topk]]}

    def predict_features(self, feats, topk=3):
        return self._result(self._probs(feats), topk)

    def predict(self, raw_all_vec, topk=3):
        return self.predict_features(self.features(raw_all_vec), topk=topk)

    def predict_robust(self, raw_all_vec, window=48, stride=12, topk=3):
        """Whole-clip prediction averaged with sliding-window votes.

        A single whole-clip pass can be tipped between two look-alike signs
        (e.g. good/help) by a few ambiguous frames.  Averaging the softmax over
        the whole clip PLUS overlapping sub-windows is a cheap ensemble that
        stabilises the winner and yields a meaningful top1-top2 `margin`, which
        the live loop uses to refuse to commit on a near-tie.
        """
        raw = np.asarray(raw_all_vec, np.float32)
        T = int(raw.shape[0])
        probs = [self._probs(self.features(raw))]
        if T > window:
            starts = list(range(0, T - window + 1, max(1, stride)))
            if starts[-1] != T - window:
                starts.append(T - window)
            for a in starts:
                probs.append(self._probs(self.features(raw[a:a + window])))
        mean = np.mean(np.stack(probs, axis=0), axis=0)
        out = self._result(mean, topk)
        out["n_windows"] = len(probs)
        return out


# ── Video -> raw (T, 1692) using the project's own extractor ──────────────────
def video_to_clip(video_path, proc_width=0):
    """Run the SAME detection/extraction the dataset was built with.
    Returns {'raw','arrays','masks','fps','T'} or None if the video won't open."""
    from types import SimpleNamespace
    sys.path.insert(0, str(_SRC / "dataprep"))
    import extract_dataset as ed
    # face=True so the (T,1692) layout matches training even though the model
    # ignores the face block — features.to_features slices by fixed offsets.
    opts = SimpleNamespace(face=True, hands=True, pose=True, smooth=True,
                           proc_width=proc_width)
    res = ed.extract_clip(str(video_path), opts)
    if res is None:
        return None
    raw, _ = ed.build_all_vector(res["arrays"], opts)
    return {"raw": raw.astype(np.float32), "arrays": res["arrays"],
            "masks": res["masks"], "fps": res["fps"], "T": res["T"]}


if __name__ == "__main__":
    from config import LANDMARKS_ALL
    p = SignPredictor()
    print(f"[predictor] backend={p.backend} classes={len(p.classes)} L={p.seq_len} D={p.in_dim}")
    sample = next(LANDMARKS_ALL.glob("*/*.npy"), None)
    if sample is not None:
        out = p.predict(np.load(sample))
        print(f"[predictor] {sample.parent.name}/{sample.stem}: "
              f"pred={out['name']} conf={out['conf']:.3f}  top3={out['topk']}")
