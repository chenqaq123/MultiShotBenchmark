"""Materialize the lowest-consistency comparisons as side-by-side images.

For every prompt entity and emergent track, all cross-appearance pairs are
already scored in the metrics details; the K lowest-similarity pairs across
the episode are exported to failure_cases/ as visual evidence (no pass/fail
judgement — these are exhibits for inspection, ranked by score).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import DEFAULTS, Proposal, save_json


def _collect_pairs(metrics: dict) -> list[dict]:
    out = []
    ps = metrics["prompt_specified"]["per_entity"]
    for ent in ps["characters"]:
        for pair in ent["face_similarity"]["pairs"]:
            out.append({"name": ent["entity_id"], "kind": "prompt_character",
                        "comparison": "face_embedding", **pair})
    for ent in ps["objects"]:
        for pair in ent["object_similarity"]["pairs"]:
            out.append({"name": ent["entity_id"], "kind": "prompt_object",
                        "comparison": "object_embedding", **pair})
    em = metrics["model_emergent"]["per_track"]
    for track in em["characters"]:
        for pair in track["similarity"]["pairs"]:
            out.append({"name": track["track_id"], "kind": "emergent_character",
                        "comparison": "face_embedding", **pair})
    for track in em["objects"]:
        for pair in track["similarity"]["pairs"]:
            out.append({"name": track["track_id"], "kind": "emergent_object",
                        "comparison": "object_embedding", **pair})
    return out


def _side_by_side(crop_a: np.ndarray, crop_b: np.ndarray, caption: str) -> np.ndarray:
    h = 256

    def fit(img):
        w = max(1, int(img.shape[1] * h / img.shape[0]))
        return cv2.resize(img, (w, h))

    a, b = fit(crop_a), fit(crop_b)
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    canvas = np.hstack([a, gap, b])
    bar = np.full((28, canvas.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, caption, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def export_failure_cases(metrics: dict, proposals: list[Proposal],
                         out_dir: str | Path, cfg: dict = DEFAULTS) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prop_by_id = {p.prop_id: p for p in proposals}
    pairs = sorted(_collect_pairs(metrics), key=lambda p: p["similarity"])
    pairs = pairs[: cfg["failure_case_top_k"]]

    manifest = []
    for rank, pair in enumerate(pairs):
        pa = prop_by_id[pair["a"][2]]
        pb = prop_by_id[pair["b"][2]]
        crop_a, crop_b = cv2.imread(pa.crop_path), cv2.imread(pb.crop_path)
        if crop_a is None or crop_b is None:
            continue
        caption = (f"{pair['name']}  shot{pair['a'][0]} vs shot{pair['b'][0]}  "
                   f"sim={pair['similarity']:.3f}")
        img = _side_by_side(crop_a, crop_b, caption)
        fname = (f"{rank:02d}_{pair['name']}_s{pair['a'][0]}"
                 f"_s{pair['b'][0]}_{pair['similarity']:.3f}.png")
        cv2.imwrite(str(out_dir / fname), img)
        manifest.append({**{k: pair[k] for k in
                            ("name", "kind", "comparison", "similarity", "a", "b")},
                         "image": fname})
    save_json(manifest, out_dir / "failure_cases.json")
    return len(manifest)
