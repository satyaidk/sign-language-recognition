"""
MediaPipe import guard.
=======================
This project uses MediaPipe's *legacy* ``mp.solutions`` API (FaceMesh, Hands,
Pose).  Google removed that API from the pip package in 0.10.31+, where
``import mediapipe as mp; mp.solutions`` raises ``AttributeError``.  Instead of
failing deep inside a stage with that cryptic error, every MediaPipe user goes
through :func:`load_solutions`, which fails early with the fix.
"""
from __future__ import annotations

PINNED_VERSION = "0.10.9"


class MediaPipeUnavailable(ImportError):
    """MediaPipe is missing or too new for the legacy solutions API."""


def load_solutions():
    """Return the ``mediapipe`` module, guaranteed to expose ``mp.solutions``."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise MediaPipeUnavailable(
            "[ERROR] mediapipe is not installed — run: pip install -r requirements.txt"
        ) from exc
    if not hasattr(mp, "solutions"):
        version = getattr(mp, "__version__", "?")
        raise MediaPipeUnavailable(
            f"[ERROR] mediapipe {version} no longer ships the legacy `mp.solutions` API "
            f"(removed in 0.10.31+). This project is pinned to mediapipe=={PINNED_VERSION}: "
            "pip install -r requirements.txt")
    return mp
