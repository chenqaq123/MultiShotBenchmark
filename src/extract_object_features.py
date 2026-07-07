"""Step 7 support: DINOv2 crop embeddings (+ HSV color histogram) for proposals."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import Proposal
from .models import dinov2_embed_images


def _pad_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.full((side, side, 3), 127, dtype=np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def color_histogram(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8],
                        [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / (hist.sum() + 1e-8)).astype(np.float32)


def extract_object_features(proposals: list[Proposal],
                            embeddings_dir: str | Path) -> dict:
    """DINOv2 embedding + HSV color histogram for every person/object crop.

    Sets proposal.dino_index; saves matrices to embeddings/proposals.npz.
    Returns feats = {"dino": {prop_id: vec}, "hist": {prop_id: vec}}.
    """
    targets = [p for p in proposals if p.kind in ("person", "object")]
    crops_rgb, hists = [], []
    for p in targets:
        bgr = cv2.imread(p.crop_path)
        hists.append(color_histogram(bgr))
        crops_rgb.append(cv2.cvtColor(_pad_square(bgr), cv2.COLOR_BGR2RGB))
    dino = dinov2_embed_images(crops_rgb)
    for i, p in enumerate(targets):
        p.dino_index = i
    hists = np.stack(hists) if hists else np.zeros((0, 1024), np.float32)
    np.savez_compressed(Path(embeddings_dir) / "proposals.npz",
                        prop_ids=np.array([p.prop_id for p in targets]),
                        dino=dino, color_hist=hists)
    return {"dino": {p.prop_id: dino[i] for i, p in enumerate(targets)},
            "hist": {p.prop_id: hists[i] for i, p in enumerate(targets)}}


def hist_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.minimum(h1, h2).sum())


def object_pair_similarity(pa: Proposal, pb: Proposal, feats: dict,
                           cfg: dict) -> float:
    """DINOv2 cosine; for two small objects, blended with color-histogram
    intersection (small crops carry little structure for DINOv2 alone)."""
    from .common import cosine
    s = cosine(feats["dino"][pa.prop_id], feats["dino"][pb.prop_id])
    if (pa.area_ratio < cfg["small_object_area_ratio"]
            and pb.area_ratio < cfg["small_object_area_ratio"]):
        w = cfg["color_hist_weight"]
        s = (1 - w) * s + w * hist_intersection(feats["hist"][pa.prop_id],
                                                feats["hist"][pb.prop_id])
    return s
