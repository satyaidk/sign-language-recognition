"""
L0 — landmark detection core.

    layout      raw landmark shapes, the (T, 1692) "all" vector layout, presence masks
    smoothing   One-Euro filter + point / hand stabilisers (pure NumPy)
    hands       anatomical handedness + hand-to-slot / hand-to-arm assignment (pure)
    extractor   persistent MediaPipe models -> per-frame landmarks (dataset AND live)
    drawing     skeleton / face / hand rendering, overlay + skeleton videos
    viewer      interactive webcam / video / image landmark viewer

Only ``extractor``, ``drawing`` and ``viewer`` import MediaPipe, so training and
most tests run without it.
"""
