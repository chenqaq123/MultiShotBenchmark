"""Step 8.1/8.2: background features per keyframe.

- DINOv2 patch tokens masked-pooled over the background mask (bg_feat)
- depth layout (Depth-Anything-V2, per-frame min-max normalized, 64x64)
- edge layout (Canny, 64x64, lightly blurred for tolerance)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import DEFAULTS, Keyframe
from .models import get_depth, get_dinov2
from .segment_foreground_background import load_bg_mask


def extract_background_features(keyframes: list[Keyframe], masks_dir: str | Path,
                                embeddings_dir: str | Path,
                                cfg: dict = DEFAULTS) -> dict[str, dict]:
    import torch

    proc, dino = get_dinov2()
    dproc, dmodel = get_depth()
    size = cfg["layout_size"]

    feats: dict[str, dict] = {}
    for kf in keyframes:
        bgr = cv2.imread(kf.path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bg_mask = load_bg_mask(masks_dir, kf.kf_id)

        # --- DINOv2 background patch pooling ---
        inputs = proc(images=rgb, return_tensors="pt").to(dino.device)
        with torch.no_grad():
            out = dino(**inputs)
        patches = out.last_hidden_state[0, 1:]                      # (N, 768)
        n = patches.shape[0]
        grid = int(round(n ** 0.5))
        mask_small = cv2.resize(bg_mask.astype(np.uint8), (grid, grid),
                                interpolation=cv2.INTER_AREA).astype(bool).ravel()
        if mask_small.sum() < max(4, 0.02 * n):
            mask_small = np.ones(n, dtype=bool)  # degenerate mask: fall back to full frame
        sel = patches[torch.from_numpy(mask_small).to(patches.device)]
        bg_feat = torch.nn.functional.normalize(sel.mean(dim=0), dim=-1).cpu().numpy()

        # --- depth layout ---
        dinputs = dproc(images=rgb, return_tensors="pt").to(dmodel.device)
        with torch.no_grad():
            dout = dmodel(**dinputs)
        depth = dout.predicted_depth[0].cpu().numpy()
        dmin, dmax = depth.min(), depth.max()
        depth = (depth - dmin) / (dmax - dmin + 1e-8)
        depth_layout = cv2.resize(depth, (size, size), interpolation=cv2.INTER_AREA)

        # --- edge layout ---
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        edge_layout = cv2.resize(edges, (size, size), interpolation=cv2.INTER_AREA)
        edge_layout = cv2.GaussianBlur(edge_layout, (5, 5), 1.5)

        feats[kf.kf_id] = {"bg_feat": bg_feat.astype(np.float32),
                           "depth_layout": depth_layout.astype(np.float32),
                           "edge_layout": edge_layout.astype(np.float32)}

    np.savez_compressed(Path(embeddings_dir) / "background.npz",
                        kf_ids=np.array(list(feats)),
                        bg_feat=np.stack([f["bg_feat"] for f in feats.values()]),
                        depth=np.stack([f["depth_layout"] for f in feats.values()]),
                        edge=np.stack([f["edge_layout"] for f in feats.values()]))
    return feats
