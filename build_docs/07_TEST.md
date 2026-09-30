# 07 — Tests: what we run, how, and the actual results

The testing rules are in `04_RULES.md` §7. This file is the **catalogue** (§3),
**how to run** it (§2), and the **append-only results log** (§4). Any metric or
performance claim must point to a row in §4.

> **Status 2026-09-30:** SignX v1 code doesn't exist yet, so every v1 test below is
> *planned*, with the task that creates it. The only tests run so far are on the
> **v0 reference code** (§4). They prove the v0 pieces we will port work.

---

## 1. Strategy: five levels

| Level | Prefix | Proves | Needs private data? | Automated |
|-------|--------|--------|---------------------|-----------|
| Unit | **UT** | Pure logic: schema, labels, splits, metrics, state machines | No | pytest |
| Component | **CT** | Stages compose correctly on synthetic data (features → train → export; build-dataset; adapters) | No (CT-12 needs a few videos) | pytest |
| Data / model | **DT** | Real datasets are valid; model quality under a named protocol | Yes | pytest `-m data/model` + reports |
| System | **ST** | End-to-end paths: throughput, latency, live ≡ file, sessions, multilingual dry run | Videos / models | Mostly |
| Real-world | **RW** | Live performance with real people and conditions | People | Scripted, manual |

## 2. How to run

> Run from the repo root; quote paths (the repo path has spaces). Use the v1 venv `.venv-signx` once T-0.2 is done.

### 2.1 Fast suite: every change, no private data
```bash
python -m pytest -q                         # after T-0.3; must pass in < 60 s
```
**Until then (v0 reference; paths change to `reference/v0/...` after T-0.1):**
```bash
python training/segmenter.py                # UT-1/UT-2 reference self-test
python training/model.py                    # v0 BiGRU shape smoke
python training/features.py                 # v0 D = 346
# Appendix A synthetic end-to-end (run from training/)
```

### 2.2 Data / model suite
```bash
python -m pytest -q -m "data or model"      # skips cleanly without resources
python -m signx qa report <dataset_version> # DT-1
python -m signx train configs/experiments/<exp>.toml     # DT-4 (val); --test for the final report
python -m signx backup verify [--root <second copy>]     # DT-8
```

### 2.3 System suite
```bash
python -m signx extract <adapter> --workers 3 --limit 50 # ST-1 throughput
python -m signx live --lang <L1> --source <video> --no-display   # ST-6 / ST-5 path
python -m signx eval-session --video <session.mp4> --script <script.txt>   # ST-5
python -m pytest -q tests/test_multilingual_dryrun.py    # ST-8
```

### 2.4 Real-world
Follow §6; log results in §4.

---

## 3. Test catalogue (v1)

### Unit
| ID | What | Pass criterion | Created by | Status |
|----|------|----------------|-----------|--------|
| UT-1 | Motion segmenter: 2 synthetic bursts → 2 onsets, 2 segments | exact | T-5.3 (port) | ✅ v0 ref passes |
| UT-2 | Debouncer: `no,no,no,yes,no(t=9)` → `no,yes,no`; disabled passes all | exact | T-5.3 (port) | ✅ v0 ref passes |
| UT-3 | Feature spec: D and the column map computed from the config | exact | T-3.1 | planned |
| UT-4 | Encoders (bigru/transformer/cnn_transformer) shapes + params ≤ 10 M | exact | T-3.4 | planned (v0 bigru ✅) |
| UT-5 | Classifier routes to the correct language head | exact | T-3.4 | planned |
| UT-6 | `paths`: unset `SIGNX_DATA_ROOT` → clear error; set → layout paths | exact | T-0.3 | planned |
| UT-7 | `config`: defaults, TOML override, unknown key → error | exact | T-0.3 | planned |
| UT-8 | Backup: write → verify OK; 1-byte corruption and a deleted file detected | exact | T-0.4 | planned |
| UT-9 | Doctor with a mocked environment → correct exit codes | exact | T-0.5 | planned |
| UT-10 | Schema `mp-1692-v1` offsets; `to_row`; float16 round-trip < 1e-3; absent → zeros + mask 0 | exact | T-1.1 | planned |
| UT-11 | One-Euro: jitter reduced on a noisy sine; step settles within N frames | thresholds | T-1.2 | planned |
| UT-12 | `Smoothed(FakeBackend)` rows as expected; no back-fill | exact | T-1.2 | planned |
| UT-13 | Handedness: mirrored vs non-mirrored input → anatomical slots | exact | T-1.3 | planned |
| UT-14 | `sign_id` format `<lang>:<GLOSS>`; unique; exactly one language | exact | T-2.1 | planned |
| UT-15 | Label map append-only; re-numbering raises | exact | T-2.1 | planned |
| UT-16 | No language code/gloss hard-coded in `signx/` (grep test) | 0 hits | T-2.1 | planned |
| UT-17 | Splits: no signer in more than one split; official splits respected | 0 leakage | T-2.2 | planned |
| UT-18 | Dataset content hash deterministic, and changes when any npz changes | exact | T-2.2 | planned |
| UT-19 | Metrics: top-k, macro-F1, per-signer, ECE vs hand-computed | exact | T-3.6 | planned |
| UT-20 | Temperature scaling lowers ECE on synthetic miscalibrated logits | ECE ↓ | T-3.8 | planned |
| UT-21 | Hand-velocity segmenter on synthetic landmark traces | exact | T-5.3 | planned |
| UT-22 | Session scorer alignment (correct/sub/ins/del) | exact | T-5.7 | planned |
| UT-23 | `predict_robust` window averaging + margin | exact | T-5.2 | planned |

### Component
| ID | What | Pass criterion | Created by | Status |
|----|------|----------------|-----------|--------|
| CT-1 | Transform output shape (L, D) | exact | T-3.1 | ✅ v0 ref |
| CT-2 | Transform deterministic | equal | T-3.1 | ✅ v0 ref |
| CT-3 | Handles T = 1 … 600 | shape | T-3.1 | ✅ v0 ref |
| CT-4 | Absent hand stays 0 after augmentation | all zero | T-3.2 | ✅ v0 ref |
| CT-5 | Flip is an involution | allclose | T-3.2 | ✅ v0 ref |
| CT-6 | Flip swaps hand slots + masks (and face-subset pairs) | as specified | T-3.2 | ✅ v0 ref (hands only) |
| CT-7 | xy translation + scale invariance | atol 1e-4 | T-3.1 | ✅ v0 ref |
| CT-8 | Tiny synthetic training fit | acc ≥ 0.9 | T-3.5 | ✅ v0 ref |
| CT-9 | ONNX parity per encoder, batch 1 and 4 | < 1e-3 | T-5.1 | ✅ v0 ref (bigru, legacy exporter) |
| CT-10 | Quantized model agreement (only if quantization is used) | ≥ 99% real | T-5.1 | v0 ref: synthetic 100% |
| CT-11 | Feature cache hit/miss + RSS bound | as specified | T-3.3 | planned |
| CT-12 | Extraction on sample videos → valid npz + provenance; re-run skips | exact | T-1.4 | planned (`data`) |
| CT-13 | QA detects synthetic corruptions | all caught | T-1.5 | planned |
| CT-14 | `build-dataset` on a synthetic source → valid manifest; refuses mixed extractor | exact | T-2.2 | planned |
| CT-15 | `folder` / `recordings` adapters on synthetic folders | exact | T-2.3 | planned |

### Data / model
| ID | What | Pass criterion | Created by | Status |
|----|------|----------------|-----------|--------|
| DT-1 | Extraction QA on the L1 dataset | FAIL rate < 2%; all FAILs reviewed | T-2.9 | planned |
| DT-2 | Dataset report: counts per class/signer/split; signers per split | matches config | T-2.9 | planned |
| DT-3 | Within-signer CV (**diagnostic only**) | — | optional | v0 historical: 0.910 |
| **DT-4** | **Signer-independent val/test**: top-1, top-5, macro-F1, per-signer, worst class | PRD §9 | T-4.1/T-4.6 | planned |
| DT-5 | Tasks backend spike comparison | table + decision | T-1.3 | planned |
| DT-6 | Processing-resolution study | decision | T-1.6 | planned |
| DT-7 | Ablation + encoder comparison (3 seeds) | tables | T-4.2/T-4.3 | planned |
| DT-8 | Backup verify on both copies + restore drill | 0 missing/mismatch | T-0.4, every milestone | planned |

### System
| ID | What | Pass criterion | Created by | Status |
|----|------|----------------|-----------|--------|
| ST-1 | Bulk extraction throughput (videos/h, frames/s, workers 1 vs N) | recorded | T-1.4 | planned |
| ST-2 | ONNX latency per encoder (CPU) | ≤ 10 ms | T-5.1 | v0 ref: 1.08–1.20 ms |
| ST-3 | Per-frame landmark latency at the serving width | ≤ 33 ms or frame-skip plan | T-1.3 | v0 legacy: ~40 ms (synthetic) |
| ST-4 | Live benchmark: fps, drops, latency p50/p95, RSS | ≥ 20 fps; p95 ≤ 700 ms; ≤ 1.5 GB | T-5.8 | planned |
| ST-5 | Session scoring on recorded sessions | reported | T-5.7 | planned |
| ST-6 | Streaming ≡ file mode | ≥ 95% same top-1 | T-5.4 | planned |
| ST-7 | 30-min soak | no crash; RSS growth < 200 MB | T-6.4 | planned |
| **ST-8** | **Multilingual dry run**: 2 synthetic languages (`qaa`, `qab`) → 2 heads → per-language report + ONNX, no code changes | passes | T-3.9 | planned |

### Real-world
| ID | What | Target (PRD §9) |
|----|------|-----------------|
| RW-1 | Scripted live sessions with new signers | ≥ 85% correct; ≤ 10% "?" |
| RW-2 | 60 s idle / non-sign behaviour | ≤ 1 false emit |
| RW-3 | Left-handed signer | ≤ 10 points below right-handed |
| RW-4 | Lighting / background variation | ≤ 10 points drop |

---

## 4. Results log (append-only; newest first)

### 2026-09-30: v0.2 refactor of the existing project (the `signlang` package)
| Test | Command | Result |
|------|---------|--------|
| Full pytest suite (unit + component + end-to-end on synthetic data, 1 test with real MediaPipe) | `python -m pytest -q` | ✅ **100 passed** in ~17 s |
| CLI smoke test: extract → verify (+overlay) → report → prune (dry-run, real, no-args rejected) → observe → train → finetune → export → video (±overlay) → live `--source` | scripted, synthetic videos + synthetic landmark dataset | ✅ 11/11 commands |
| Live classification time after a sign closes | session log `classify_ms` | ~2 ms (was ~2.5 s of re-extraction for a 2 s sign) |
| Streaming landmark extraction, persistent models, 720p→480 px | timed on synthetic frames | ~25 ms/frame (face on 26.0, off 24.7; no real face in frame) |
| Bugs found by the new tests | — | verifier crash on frame-count mismatch; feature transform crash on a 0-frame clip (both fixed, regression-tested) |

### 2026-09-30: direction change (no new test runs)
The owner confirmed that v0 data and weights are permanently lost and that v1 will be a new, language-agnostic platform. No v1 code yet. The v0 results below remain valid as **evidence that the reference components work**.

### 2026-09-30: v0 reference verification (reference laptop: i5-11320H, 7.8 GB, Win 11, Python 3.11.0, torch 2.11.0+cpu)

| Test | Command | Result |
|------|---------|--------|
| UT-1, UT-2 (v0) | `python training/segmenter.py` | ✅ `starts=2 segments_closed=2 lengths=[36, 31]`; dedup `['no','no','no','yes','no'] -> ['no','yes','no']`; `[self-test] OK` |
| UT-4 (v0) | `python training/model.py` | ✅ `arch=bigru params=543,553 in (4,64,346) -> out (4, 12)` |
| UT-3 (v0) | `python training/features.py` | ✅ `n_points=55 coord_dim=165 n_vis=13 n_mask=3`, D = 346 |
| UT-4 (v0 transformer) | ad hoc | ✅ params 836,289; out (2, 12) |
| v0 `data.py` | `python training/data.py` | ❌ `FileNotFoundError` (v0 data lost; expected) |
| CT-1…CT-10 (v0) | Appendix A | ✅ **ALL PASS**, twice (incl. once extracted verbatim from this file): parity 2.4e-6 / 2.9e-6; int8 agreement 100%; tiny-train acc 1.00 in 4.2 s |
| ST-2 (v0) | Appendix A | fp32 ONNX **1.08–1.20 ms/clip**, int8 1.07 ms; sizes 2,120 KB vs 1,993 KB (GRU not quantized) |
| — | v0 feature transform | ~1.41 ms/clip (T = 90) |
| ST-3 (v0 legacy) | `extract_clip`, 45-frame 1080p **noise** video, face+hands+pose | 3-model init+close 97 ms; **~40 ms/frame** at full width and at 480 px. No person in frame, so re-measure on real video in T-1.3 |
| CT-9 exporter | legacy under torch 2.11 | ✅ works with `DeprecationWarning … legacy TorchScript-based ONNX export` + `The feature will be removed`; also a warning about variable-batch GRU (parity at batch 3/5 passed) |
| CT-9 dynamo | `dynamo=True` | ❌ `ModuleNotFoundError: onnxscript` (not installed; untested) |

### Historical v0 results (June–July 2026; data now lost)

| Test | Date | Result | Source |
|------|------|--------|--------|
| v0 extraction QA | 2026-06-24 | 134 PASS; detection: pose 0.994, face 0.939, left hand 0.222, right hand 0.861 | `reference/v0/processed_dataset/extraction_report.json` |
| v0 within-signer CV | ~2026-06-25 | OOF acc 0.9104, macro-F1 0.9091; folds ±0.029 (**one signer; mixed ASL/ISL labels; diagnostic only**) | `…/training/artifacts/metrics/cv_report.json` |
| v0 ONNX parity | 2026-06-25 | 1.67e-6 | `…/exported/model_meta.json` |
| v0 live session | 2026-06-27 | 28 emits, no ground truth (can't be scored) | `…/predictions/live_transcript.json` |

---

## 5. Known gaps

1. No v1 code, so no v1 tests yet (they arrive with their tasks).
2. The MediaPipe Tasks backend is untested on this machine (T-1.3).
3. Real-video landmark latency is unknown (only a synthetic legacy measurement exists).
4. The `dynamo` ONNX exporter is untested (needs `onnxscript`).
5. No signer-independent number exists for any model yet.

---

## 6. Real-world test protocol (RW)

**Participants:** ≥ 5 people not in training data; ≥ 1 left-handed; ≥ 2 Deaf or fluent L1 signers; consent recorded.
**Setup:** reference laptop; webcam 720p/30 fps; 1–2 m distance; upper body + hands in frame; two lighting setups (bright even, dim/side) and two backgrounds (plain, busy); `signx live --lang <L1>`.

**Per participant (~15 min):**
1. Warm-up, 1 min (not scored).
2. **RW-1:** 3 scripted sequences of 12–20 signs from the v1 vocabulary (shuffled), natural pauses.
3. **RW-2:** 60 s of rest, talking, face-touching, adjusting glasses/hair, reaching off-screen. Every emit counts as a false positive.
4. **RW-4:** repeat one sequence in the other lighting/background.
5. Feedback: usable? which signs failed? what would make it useful?

**Scoring:** `signx eval-session` aligns the emitted transcript with the script → correct / substitution / deletion / insertion; "?" counted separately; latency from the session logs.

**Results template (copy into §4):**

| Date | Model version | Lang | Participant (pseudonym) | Hand | Signer type | Condition | Script len | Correct | Sub | Del | Ins | "?" | Accuracy | Idle false emits/60 s | Latency p50/p95 | Notes |
|------|---------------|------|------------------------|------|-------------|-----------|-----------|---------|-----|-----|-----|-----|----------|-----------------------|-----------------|-------|

## 7. Acceptance thresholds

See PRD §9. Summary: signer-independent top-1 ≥ 0.85 on the 100 ASL signs (ASL Citizen official test signers); top-5 ≥ 0.95; RW-1 ≥ 85% with ≤ 10% "?"; RW-2 ≤ 1/60 s; p95 ≤ 700 ms; ≥ 20 fps; left-handed gap ≤ 10 pts; ST-8 passes.

## 8. Regression policy

- The fast suite must pass before any task is `DONE`.
- Changes to features, augmentation, encoders, or training → re-run DT-4 on val (same dataset version, ≥ 3 seeds for claims).
- A detection/backend/smoothing/resolution change → new `extractor_id` → new dataset version → re-run DT-1/DT-2/DT-4.
- A torch or onnxruntime change → CT-9 + ST-2.
- A live-loop change → UT-1/UT-2/UT-21 + ST-6 + ST-5 on the recorded sessions.

---

## Appendix A — v0 synthetic end-to-end script (source for porting CT-1…CT-10 in T-3.1/T-3.2/T-3.5/T-5.1)

Run from v0's `training/` folder (`reference/v0/training/` after T-0.1). It needs no data. Verified 2026-09-30: ALL PASS.

```python
"""Synthetic end-to-end check: raw (T,1692) -> features -> augment -> train -> ONNX export -> parity."""
import copy, sys, time, warnings, tempfile, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.getcwd())
import features as F
from config import CONFIG, LEFT_SHOULDER, RIGHT_SHOULDER
from data import augment, SignDataset
from model import build_model

rng = np.random.default_rng(0)
fc, tc = CONFIG.feature, CONFIG.train
spec = F.build_spec(fc)

def fake_clip(T, right_only=True):
    raw = np.zeros((T, 1692), np.float32)
    pose = rng.uniform(0.3, 0.7, (T, 33, 4)).astype(np.float32); pose[..., 3] = 0.9
    pose[:, LEFT_SHOULDER, :2] = [0.6, 0.5]; pose[:, RIGHT_SHOULDER, :2] = [0.4, 0.5]
    raw[:, 0:132] = pose.reshape(T, -1)
    raw[:, 132:1566] = rng.uniform(0.4, 0.6, (T, 1434))
    raw[:, 1629:1692] = rng.uniform(0.3, 0.7, (T, 63))     # right hand only
    if not right_only:
        raw[:, 1566:1629] = rng.uniform(0.3, 0.7, (T, 63))
    return raw

results = {}
c = fake_clip(60); a = F.to_features(c, fc, spec); b = F.to_features(c, fc, spec)
results["CT-1 shape_(64,346)"] = a.shape == (64, 346)
results["CT-2 deterministic"] = np.array_equal(a, b)
results["CT-3 handles_T=1,14,286,600"] = all(F.to_features(fake_clip(T), fc, spec).shape == (64, 346) for T in (1, 14, 286, 600))
c2 = c.copy()                                   # CT-7: scale 1.5x + shift 0.1 in x/y of present points
for s, e, ch in ((0, 132, 4), (132, 1566, 3), (1566, 1692, 3)):
    blk = c2[:, s:e].reshape(60, -1, ch); nz = np.abs(blk[..., :3]).sum(-1, keepdims=True) > 0
    blk[..., :2] = np.where(nz, blk[..., :2] * 1.5 + 0.1, 0); c2[:, s:e] = blk.reshape(60, -1)
f2 = F.to_features(c2, fc, spec)
results["CT-7 translation+scale_invariant(xy)"] = np.allclose(
    a[:, :165].reshape(64, 55, 3)[..., :2], f2[:, :165].reshape(64, 55, 3)[..., :2], atol=1e-4)
co, vi, pr = F.geometric(c, fc, spec)
ok = True
for i in range(50):                              # CT-4
    tc2 = copy.deepcopy(tc); tc2.aug_flip_prob = 0.0
    co2, _, pr2 = augment(co, vi, pr, tc2, spec, np.random.default_rng(i))
    s, e = spec.seg["left_hand"]; ok &= bool(np.all(co2[:, s:e] == 0))
results["CT-4 absent_hand_stays_zero_after_aug"] = ok
cf = co[:, spec.point_perm].copy(); cf[..., 0] *= -1; cf2 = cf[:, spec.point_perm].copy(); cf2[..., 0] *= -1
results["CT-5 flip_is_involution"] = np.allclose(co, cf2)
tc_flip = copy.deepcopy(tc); tc_flip.aug_flip_prob = 1.0
cff, _, prf = augment(co, vi, pr, tc_flip, spec, np.random.default_rng(1))
s, e = spec.seg["left_hand"]
results["CT-6 flip_swaps_hand_slots"] = bool(np.abs(cff[:, s:e]).sum() > 0 and prf["left_hand"].sum() > 0)
clips, labels = [], []                            # CT-8: 3 separable synthetic classes
for k in range(3):
    for j in range(12):
        raw = fake_clip(int(rng.integers(30, 90))); raw[:, 1629:1692] += k * 0.1
        clips.append(F.geometric(raw, fc, spec)); labels.append(k)
labels = np.array(labels)
ds = SignDataset(clips, labels, fc, spec, tc, augment_on=True)
m = build_model(346, 3, CONFIG.model); opt = torch.optim.AdamW(m.parameters(), 2e-3)
dl = torch.utils.data.DataLoader(ds, batch_size=12, shuffle=True); lossf = torch.nn.CrossEntropyLoss()
for ep in range(15):
    m.train()
    for x, y in dl:
        opt.zero_grad(); l = lossf(m(x), y); l.backward(); opt.step()
m.eval()
X = torch.stack([torch.from_numpy(F.assemble(*cl, fc, spec)) for cl in clips])
acc = float((m(X).argmax(1).numpy() == labels).mean())
results[f"CT-8 tiny_train_fits_synthetic(acc={acc:.2f})"] = acc >= 0.9
import onnxruntime as ort                         # CT-9 / CT-10 / ST-2
p = Path(tempfile.gettempdir()) / "synthetic_sign.onnx"
torch.onnx.export(m, torch.randn(1, 64, 346), str(p), dynamo=False, input_names=["features"],
                  output_names=["logits"], dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17, do_constant_folding=True)
sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
xb = X[:5].numpy()
diff = float(np.abs(sess.run(["logits"], {"features": xb})[0] - m(X[:5]).detach().numpy()).max())
results[f"CT-9 onnx_parity(max_diff={diff:.1e})"] = diff < 1e-3
from onnxruntime.quantization import quantize_dynamic, QuantType
p8 = p.with_suffix(".int8.onnx"); quantize_dynamic(str(p), str(p8), weight_type=QuantType.QInt8)
s8 = ort.InferenceSession(str(p8), providers=["CPUExecutionProvider"])
agree = float((s8.run(["logits"], {"features": xb})[0].argmax(1)
               == sess.run(["logits"], {"features": xb})[0].argmax(1)).mean())
results[f"CT-10 int8_top1_agreement({agree:.0%})"] = agree >= 0.8
x1 = xb[:1]; [sess.run(["logits"], {"features": x1}) for _ in range(5)]
t0 = time.perf_counter(); [sess.run(["logits"], {"features": x1}) for _ in range(200)]
print(f"ST-2 latency fp32 {(time.perf_counter() - t0) / 200 * 1000:.2f} ms/clip; "
      f"sizes fp32 {p.stat().st_size // 1024} KB int8 {p8.stat().st_size // 1024} KB")
for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k)
print("ALL PASS" if all(results.values()) else "SOME FAILED")
```
