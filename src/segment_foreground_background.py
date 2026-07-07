"""Step 4: foreground / background separation.

Foreground = union of person + object proposal masks (background-region
proposals stay in the background). Background mask is the complement; no
inpainting — downstream background features use masked pooling.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import Keyframe, Proposal
from .extract_open_world_proposals import load_proposal_masks


def segment_foreground_background(keyframes: list[Keyframe], proposals: list[Proposal],
                                  masks_dir: str | Path) -> dict[str, dict]:
    """Writes <kf_id>_fg.png / <kf_id>_bg.png; returns per-kf mask info."""
    masks_dir = Path(masks_dir)
    by_kf: dict[str, list[Proposal]] = {}
    for p in proposals:
        by_kf.setdefault(p.kf_id, []).append(p)

    info: dict[str, dict] = {}
    for kf in keyframes:
        fg = np.zeros((kf.height, kf.width), dtype=bool)
        prop_masks = load_proposal_masks(masks_dir, kf.kf_id)
        for p in by_kf.get(kf.kf_id, []):
            if p.kind in ("person", "object") and p.prop_id in prop_masks:
                fg |= prop_masks[p.prop_id]
        bg = ~fg
        bg_ratio = float(bg.mean())
        kf.bg_visible_ratio = round(bg_ratio, 4)
        cv2.imwrite(str(masks_dir / f"{kf.kf_id}_fg.png"), fg.astype(np.uint8) * 255)
        cv2.imwrite(str(masks_dir / f"{kf.kf_id}_bg.png"), bg.astype(np.uint8) * 255)
        info[kf.kf_id] = {"bg_visible_ratio": bg_ratio,
                          "fg_mask": str(masks_dir / f"{kf.kf_id}_fg.png"),
                          "bg_mask": str(masks_dir / f"{kf.kf_id}_bg.png")}
    return info


def load_bg_mask(masks_dir: str | Path, kf_id: str) -> np.ndarray:
    m = cv2.imread(str(Path(masks_dir) / f"{kf_id}_bg.png"), cv2.IMREAD_GRAYSCALE)
    return m > 127
