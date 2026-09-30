"""
Temporal smoothing for landmark streams (pure NumPy, no MediaPipe).
==================================================================
Raw MediaPipe output jitters frame to frame, which makes skeletons flicker and
adds noise to the velocity features.  The One-Euro filter (Casiez et al., 2012)
fixes that with almost no lag: it smooths hard while a point is nearly still
and barely filters while it moves fast.

    OneEuroFilter     element-wise filter over a scalar or an array of values
    PointStabilizer   a fixed set of points (pose / one hand): One-Euro on x/y,
                      teleport reset, EMA-smoothed visibility, optional
                      occlusion hold (used by the viewer, OFF for datasets)
    HandStabilizer    one PointStabilizer per tracked hand, keyed by label

Points are ``(N, C)`` arrays with C = 3 (x, y, z) or 4 (+ visibility).
"""
from __future__ import annotations

import math

import numpy as np

# One-Euro parameters tuned for normalised image coordinates at ~30 fps.
POSE_EURO = dict(min_cutoff=1.2, beta=0.30, d_cutoff=1.0)
HAND_EURO = dict(min_cutoff=2.0, beta=0.80, d_cutoff=1.0)
RESET_DIST_NORM = 0.30   # a jump larger than this (normalised units) resets a point


class OneEuroFilter:
    """Adaptive low-pass filter; works element-wise on scalars or arrays."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float64))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x.copy()
        dt = t - self.t_prev
        if dt <= 0.0:                      # duplicate / out-of-order timestamp
            dt = 1e-3
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()

    def reset_where(self, mask: np.ndarray, x) -> None:
        """Restart the filter for the masked elements at value ``x`` (teleport)."""
        if self.x_prev is None:
            return
        x = np.asarray(x, dtype=np.float64)
        self.x_prev = np.where(mask, x, self.x_prev)
        self.dx_prev = np.where(mask, 0.0, self.dx_prev)


class PointStabilizer:
    """Smooths a fixed-size point set; optionally holds it through short dropouts.

    ``update(points, t) -> (smoothed | None, stale)``.  ``stale`` is True when the
    returned points are a *held* copy from an earlier frame (occlusion hold).
    """

    def __init__(self, n: int, euro: dict, hold_frames: int = 0,
                 reset_dist: float = RESET_DIST_NORM):
        self.n = int(n)
        self.hold_frames = int(hold_frames)
        self.reset_dist = float(reset_dist)
        self.filter = OneEuroFilter(**euro)      # filters the (n, 2) x/y block at once
        self.vis = None                          # EMA of visibility (C == 4 only)
        self.last = None
        self.miss = 0

    def reset(self) -> None:
        self.filter.reset()
        self.vis = None
        self.last = None
        self.miss = 0

    def update(self, points, t: float):
        if points is None:
            self.miss += 1
            if self.last is not None and self.miss <= self.hold_frames:
                return self.last, True
            self.reset()
            return None, False

        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[0] != self.n:
            raise ValueError(f"expected {self.n} points, got {pts.shape[0]}")
        xy = pts[:, :2].astype(np.float64)

        # Reset points that teleported so the filter doesn't smear across a jump.
        if self.reset_dist and self.filter.x_prev is not None:
            jump = np.hypot(*(xy - self.filter.x_prev).T) > self.reset_dist
            if jump.any():
                self.filter.reset_where(jump[:, None], xy)

        out = pts.copy()
        out[:, :2] = self.filter(xy, t)
        if pts.shape[1] >= 4:                    # smooth visibility with an EMA
            v = pts[:, 3]
            self.vis = v.copy() if self.vis is None else 0.6 * self.vis + 0.4 * v
            out[:, 3] = self.vis

        self.last = out
        self.miss = 0
        return out, False


class HandStabilizer:
    """One PointStabilizer per hand, keyed by MediaPipe's handedness label.

    A hand that is not seen in a frame has its filter dropped, so when it comes
    back it starts fresh instead of being blended with where it was seconds ago.
    """

    def __init__(self, euro: dict = HAND_EURO):
        self.euro = euro
        self._stabs: dict[str, PointStabilizer] = {}

    def reset(self) -> None:
        self._stabs.clear()

    def update(self, hands, t: float):
        """``hands``: list of ``(points (21, 3), raw_label)`` -> same, smoothed."""
        out, seen, counts = [], set(), {}
        for points, raw in hands:
            counts[raw] = counts.get(raw, 0) + 1
            key = raw if counts[raw] == 1 else f"{raw}#{counts[raw]}"   # duplicate labels
            seen.add(key)
            stab = self._stabs.get(key)
            if stab is None:
                stab = self._stabs[key] = PointStabilizer(len(points), self.euro, hold_frames=0)
            smoothed, _ = stab.update(points, t)
            out.append((smoothed, raw))
        for key in list(self._stabs):
            if key not in seen:
                del self._stabs[key]
        return out
