"""
Hand side logic: which detected hand is the signer's LEFT and which is RIGHT.
============================================================================
MediaPipe Hands reports handedness assuming a **mirrored** (selfie) image.

  * Live webcam shown mirrored  -> the raw label is already the true side.
  * Raw video / non-mirrored    -> the label is swapped to get the true side.

That label is occasionally wrong — the classic failure is both hands coming back
labelled "Left".  The old extractor then wrote both into slot 0 and silently
lost one hand.  ``assign_hand_slots`` resolves such duplicates geometrically:
by the nearest pose wrist when the body is tracked, else by image position.
"""
from __future__ import annotations

import math

import numpy as np

from signlang.landmarks.layout import LEFT_WRIST, RIGHT_WRIST

LEFT_SLOT, RIGHT_SLOT = 0, 1


def anatomical_label(raw_label: str, mirrored: bool) -> str:
    """Map MediaPipe's handedness label to the signer's true side."""
    if mirrored:
        return raw_label
    return "Right" if raw_label == "Left" else "Left"


def match_hands_to_sides(hand_wrists, lw, rw) -> dict:
    """Assign each hand wrist to a body side by proximity to the pose wrists.

    For two hands an optimal pairing is used so the drawn arms never cross (X).
    Points are any 2-D coordinates (pixels or normalised).  Returns
    ``{hand_index: "Left" | "Right"}``.
    """
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    n = len(hand_wrists)
    if n == 0:
        return {}
    if n == 1:
        return {0: "Left" if d(hand_wrists[0], lw) <= d(hand_wrists[0], rw) else "Right"}

    h0, h1 = hand_wrists[0], hand_wrists[1]
    if d(h0, lw) + d(h1, rw) <= d(h0, rw) + d(h1, lw):
        sides = {0: "Left", 1: "Right"}
    else:
        sides = {0: "Right", 1: "Left"}
    for i in range(2, n):
        sides[i] = "Left" if d(hand_wrists[i], lw) <= d(hand_wrists[i], rw) else "Right"
    return sides


def _pose_wrists_usable(pose, vis_thresh: float = 0.5) -> bool:
    if pose is None:
        return False
    pose = np.asarray(pose)
    if pose.shape[1] < 4:
        return True
    return bool(pose[LEFT_WRIST, 3] >= vis_thresh and pose[RIGHT_WRIST, 3] >= vis_thresh)


def assign_hand_slots(hands, pose=None, mirrored: bool = False) -> list:
    """Decide the anatomical slot of every detected hand.

    ``hands``: list of ``(points (21, >=2), raw_label)``; ``pose``: (33, >=2)
    pose landmarks of the same frame or ``None``.
    Returns ``[(slot, points), ...]`` with at most one hand per slot.
    """
    if not hands:
        return []
    sides = [anatomical_label(label, mirrored) for _, label in hands]

    if len(hands) >= 2 and len(set(sides[:2])) == 1:
        # Duplicate handedness: resolve the two hands geometrically.
        wrists = [np.asarray(p)[0, :2] for p, _ in hands[:2]]
        if not mirrored and _pose_wrists_usable(pose):
            pose = np.asarray(pose)
            by_idx = match_hands_to_sides(wrists, pose[LEFT_WRIST, :2], pose[RIGHT_WRIST, :2])
            sides[:2] = [by_idx[0], by_idx[1]]
        else:
            # In a non-mirrored frame the signer's RIGHT hand appears on the
            # image's left (smaller x); a mirrored frame is the other way round.
            right_first = wrists[0][0] < wrists[1][0]
            if mirrored:
                right_first = not right_first
            sides[:2] = ["Right", "Left"] if right_first else ["Left", "Right"]

    out, used = [], set()
    for (points, _), side in zip(hands, sides):
        slot = LEFT_SLOT if side == "Left" else RIGHT_SLOT
        if slot in used:                 # >2 hands (not expected): keep the first
            continue
        used.add(slot)
        out.append((slot, points))
    return out
