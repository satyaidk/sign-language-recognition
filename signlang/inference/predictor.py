"""
Shared inference: load the exported model and turn landmarks into a sign.
========================================================================
Used by BOTH file recognition (video.py) and live recognition (live.py), so the
serving path is identical everywhere.

``SignPredictor`` is self-describing: it reads ``model_meta.json`` (written by
export / finetune), rebuilds the EXACT FeatureConfig used in training — never
the current defaults — and runs the ONNX model (default, fast on CPU) or the
PyTorch checkpoint.

``extractor_options()`` tells the landmark extractor what the model needs: if the
model ignores the face, face detection is skipped entirely (the face block stays
zero, which the feature transform ignores anyway) — a large per-frame saving.
"""
from __future__ import annotations

import numpy as np

from signlang.config import Paths, feature_config_from_dict, get_paths, model_config_from_dict
from signlang.training import features as F
from signlang.utils import load_json


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class SignPredictor:
    def __init__(self, paths: Paths | None = None, backend: str = "onnx", int8: bool = False):
        paths = paths or get_paths()
        meta_path = paths.exported / "model_meta.json"
        if backend == "onnx" and not meta_path.exists():
            print("[predictor] no exported ONNX model found — falling back to the PyTorch checkpoint")
            backend = "torch"
        if backend == "torch":
            meta_path = paths.checkpoints / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — train a model first "
                "(python -m signlang train / finetune / export)")

        self.meta = load_json(meta_path)
        self.classes = self.meta["classes"]
        self.seq_len = self.meta["seq_len"]
        self.in_dim = self.meta["in_dim"]
        self.fcfg = feature_config_from_dict(self.meta["config"]["feature"])
        self.spec = F.build_spec(self.fcfg)
        self.backend = backend

        if backend == "onnx":
            import onnxruntime as ort
            name = (self.meta.get("onnx_int8") if int8 else None) or self.meta.get("onnx") or "sign_model.onnx"
            if int8 and not self.meta.get("onnx_int8"):
                print("[predictor] no int8 model was exported — using fp32")
            self.sess = ort.InferenceSession(str(paths.exported / name), providers=["CPUExecutionProvider"])
        elif backend == "torch":
            import torch
            from signlang.training.model import build_model
            ck = torch.load(paths.checkpoints / "model_final.pt", map_location="cpu", weights_only=False)
            self.torch = torch
            self.model = build_model(ck["in_dim"], ck["n_classes"], model_config_from_dict(ck["config"]["model"]),
                                     max_len=ck["seq_len"]).eval()
            self.model.load_state_dict(ck["state_dict"])
        else:
            raise ValueError(f"unknown backend: {backend!r} (use 'onnx' or 'torch')")

    # ── what the model needs from the landmark extractor ──
    def extractor_options(self, proc_width: int = 0, smooth: bool = True):
        from signlang.landmarks.extractor import ExtractorOptions
        return ExtractorOptions(face=self.fcfg.use_face, hands=True, pose=True,
                                smooth=smooth, proc_width=proc_width)

    # ── features + logits ──
    def features(self, raw_all_vec: np.ndarray) -> np.ndarray:
        return F.to_features(raw_all_vec, self.fcfg, self.spec)

    def _logits(self, feats: np.ndarray) -> np.ndarray:
        x = feats[None].astype(np.float32)
        if self.backend == "onnx":
            return self.sess.run(["logits"], {"features": x})[0][0]
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x)).numpy()[0]

    def _probs(self, feats: np.ndarray) -> np.ndarray:
        return softmax(self._logits(feats))

    def _result(self, probs: np.ndarray, topk: int) -> dict:
        order = np.argsort(probs)[::-1]
        i0 = int(order[0])
        i1 = int(order[1]) if probs.shape[0] > 1 else i0
        return {"label": i0, "name": self.classes[i0], "conf": float(probs[i0]),
                "margin": float(probs[i0] - probs[i1]) if i1 != i0 else float(probs[i0]),
                "probs": probs,
                "topk": [(self.classes[int(i)], float(probs[int(i)])) for i in order[:topk]]}

    def predict_features(self, feats: np.ndarray, topk: int = 3) -> dict:
        return self._result(self._probs(feats), topk)

    def predict(self, raw_all_vec: np.ndarray, topk: int = 3) -> dict:
        """Whole-clip prediction for a raw (T, 1692) landmark sequence."""
        return self.predict_features(self.features(raw_all_vec), topk=topk)

    def predict_robust(self, raw_all_vec: np.ndarray, window: int = 48, stride: int = 12, topk: int = 3) -> dict:
        """Whole-clip prediction averaged with sliding-window votes.

        One whole-clip pass can be tipped between two look-alike signs (e.g.
        good / help) by a few ambiguous frames.  Averaging the softmax over the
        whole clip PLUS overlapping sub-windows is a cheap ensemble that
        stabilises the winner and gives a meaningful top1 - top2 ``margin``,
        which live mode uses to refuse to commit on a near-tie.
        """
        raw = np.asarray(raw_all_vec, np.float32)
        T = int(raw.shape[0])
        probs = [self._probs(self.features(raw))]
        if T > window:
            starts = list(range(0, T - window + 1, max(1, stride)))
            if starts[-1] != T - window:
                starts.append(T - window)
            probs += [self._probs(self.features(raw[a:a + window])) for a in starts]
        out = self._result(np.mean(np.stack(probs, axis=0), axis=0), topk)
        out["n_windows"] = len(probs)
        return out


def load_predictor(paths: Paths | None = None, backend: str = "onnx", int8: bool = False) -> SignPredictor:
    """Construct a predictor, turning a missing model into a clean CLI error."""
    try:
        return SignPredictor(paths, backend=backend, int8=int8)
    except FileNotFoundError as exc:
        raise SystemExit(f"[ERROR] {exc}")
