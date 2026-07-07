"""Steps 8.1–8.4: pairwise same-view scores between keyframes.

same_view_score = 0.60 * S_dino_bg + 0.25 * S_depth + 0.15 * S_edge
(the no-geometry variant of pipeline_plan §15.1), each component p5/p95
normalized within the episode. Frames with too little visible background
are excluded (pipeline_plan §12.2).
"""
from __future__ import annotations

import numpy as np

from .common import DEFAULTS, Keyframe, cosine, p5_p95_normalize, ssim


def compute_same_view_scores(keyframes: list[Keyframe], bg_feats: dict[str, dict],
                             cfg: dict = DEFAULTS) -> dict:
    eligible = [kf for kf in keyframes
                if (kf.bg_visible_ratio or 0) >= cfg["bg_min_visible_ratio"]
                and kf.kf_id in bg_feats]
    excluded = [kf.kf_id for kf in keyframes if kf not in eligible]
    ids = [kf.kf_id for kf in eligible]
    n = len(ids)
    raw = {"dino_bg": np.eye(n), "depth": np.eye(n), "edge": np.eye(n)}
    for i in range(n):
        fi = bg_feats[ids[i]]
        for j in range(i + 1, n):
            fj = bg_feats[ids[j]]
            raw["dino_bg"][i, j] = raw["dino_bg"][j, i] = cosine(fi["bg_feat"], fj["bg_feat"])
            raw["depth"][i, j] = raw["depth"][j, i] = ssim(fi["depth_layout"], fj["depth_layout"])
            raw["edge"][i, j] = raw["edge"][j, i] = ssim(fi["edge_layout"], fj["edge_layout"])

    combined = np.eye(n)
    norm = {}
    if n >= 2:
        iu = np.triu_indices(n, k=1)
        combined = np.zeros((n, n))
        for name, w in cfg["same_view_weights"].items():
            vals = p5_p95_normalize(raw[name][iu])
            m = np.eye(n)
            m[iu] = vals
            m.T[iu] = vals
            norm[name] = m
            combined += w * m
        np.fill_diagonal(combined, 1.0)

    return {"kf_ids": ids, "excluded_kf_ids": excluded,
            "raw": raw, "normalized": norm, "combined": combined}
