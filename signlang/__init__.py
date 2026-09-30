"""
signlang — isolated sign-language recognition from MediaPipe landmarks.

Layers (each a sub-package):

    signlang.landmarks   L0  MediaPipe detection, One-Euro smoothing, drawing, live viewer
    signlang.dataset     L1  video dataset -> verified landmark dataset (+ QA, reports)
    signlang.training    L2  features, augmentation, model, k-fold CV, final fit, ONNX export
    signlang.inference   L3  shared predictor, motion segmenter, file + live recognition

Run any stage through the CLI:  ``python -m signlang --help``.
"""

__version__ = "0.2.0"
