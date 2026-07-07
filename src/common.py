"""Shared utilities, config and data structures for the evaluation suite."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default hyper-parameters (single source of truth, see pipeline_plan.md)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # keyframe extraction
    "kf_positions": [0.15, 0.5, 0.85],      # first-stable / middle / last-stable
    "kf_search_window": 5,                   # frames around each position to pick sharpest
    "kf_min_blur": 40.0,                     # Laplacian variance floor
    "kf_min_brightness": 20.0,               # mean gray floor (black frame)
    "kf_max_brightness": 235.0,              # mean gray ceiling (white frame)
    "kf_min_contrast": 15.0,                 # gray std floor (flat / fade frame)
    "kf_min_gap_sec": 0.4,                   # dedupe keyframes closer than this
    # shot detection (always algorithmic; never taken from prompt/config)
    "shot_adaptive_k": 8.0,                  # robust z-score (MAD units) to flag a cut
    "shot_min_diff": 0.04,                   # absolute floor on 1 - hist correlation
    "shot_min_pix_diff": 0.03,               # absolute floor on mean gray pixel diff
    "shot_verify_gap": 3,                    # frames each side for cross-gap verification
    "shot_min_len_sec": 0.5,
    # proposals
    "det_box_threshold": 0.30,
    "det_text_threshold": 0.25,
    "prop_min_area_ratio": 0.002,
    "prop_max_area_ratio": 0.90,
    "prop_nms_iou": 0.85,
    "prop_min_blur": 15.0,
    # background
    "bg_min_visible_ratio": 0.15,            # below this, frame skips same-view grouping
    # association / clustering
    "face_match_threshold": 0.35,            # cos sim: same identity (pass rate)
    "face_cluster_threshold": 0.45,          # emergent person clustering
    "object_match_threshold": 0.60,          # cos sim: same object (pass rate)
    "object_cluster_threshold": 0.70,        # emergent object clustering
    "frag_margin": 0.05,                     # near-threshold centroid sim => fragmentation
    "assoc_continuity_weight": 0.5,          # embedding-continuity share in association
    "small_object_area_ratio": 0.02,         # below this, blend color histogram
    "color_hist_weight": 0.3,                # histogram share for small-object similarity
    # failure case export
    "failure_case_top_k": 5,                 # lowest-consistency pairs to materialize
    # medoid selection
    "medoid_tolerance": 0.98,                # candidates within this factor of best mean sim
    # same-view grouping
    "same_view_weights": {"dino_bg": 0.60, "depth": 0.25, "edge": 0.15},
    "same_view_knn": 3,
    "same_view_threshold": 0.55,             # on normalized combined score
    "same_view_raw_dino_floor": 0.55,        # absolute floor on raw bg similarity
    # layout
    "layout_size": 64,
    "dino_model": "facebook/dinov2-base",
    "gdino_model": "IDEA-Research/grounding-dino-tiny",
    "sam_model": "facebook/sam-vit-base",
    "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
    # generic open-world vocabulary for model-emergent discovery
    "open_world_vocab": [
        "person", "face", "chair", "table", "lamp", "plate", "bowl", "cup",
        "glass", "bottle", "basket", "spoon", "fork", "knife", "food",
        "bread", "picture frame", "window", "door", "curtain", "plant",
        "bag", "phone", "book", "hat", "watch",
    ],
}


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int          # exclusive
    start_sec: float
    end_sec: float


@dataclass
class Keyframe:
    kf_id: str
    shot_index: int
    frame_idx: int
    time_sec: float
    path: str
    width: int
    height: int
    blur: float
    brightness: float
    bg_visible_ratio: float | None = None


@dataclass
class Proposal:
    prop_id: str
    kf_id: str
    shot_index: int
    box: list            # [x1, y1, x2, y2] float, pixel coords
    score: float
    label: str           # text query that fired
    kind: str            # "person" | "object" | "background-region"
    area_ratio: float
    crop_path: str = ""
    from_entity_query: str | None = None   # prompt entity id if query came from it
    assigned_entity: str | None = None     # entity id after association
    track_id: str | None = None            # emergent track id after clustering
    face_embedding: list | None = None     # set by face stage (persons only)
    face_det_score: float | None = None
    dino_index: int | None = None          # row in object embedding matrix


def load_episode_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("entities", {"characters": [], "objects": [], "backgrounds": []})
    for key in ("characters", "objects", "backgrounds"):
        cfg["entities"].setdefault(key, [])
    base = Path(path).resolve().parent
    video = Path(cfg["video"])
    if not video.is_absolute():
        cfg["video"] = str((base / video).resolve())
    return cfg


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_np_default),
                    encoding="utf-8")


def _np_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def p5_p95_normalize(values: np.ndarray) -> np.ndarray:
    """S_norm = clip((S - p5) / (p95 - p5), 0, 1)  (pipeline_plan §14.2)."""
    v = np.asarray(values, dtype=np.float32)
    p5, p95 = np.percentile(v, 5), np.percentile(v, 95)
    if p95 - p5 < 1e-6:
        return np.clip(v - p5 + 0.5, 0.0, 1.0)
    return np.clip((v - p5) / (p95 - p5), 0.0, 1.0)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM between two same-size float images normalized to [0,1]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                 / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


def laplacian_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def box_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if a1 + a2 - inter > 0 else 0.0


def pick_device():
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}"


def ensure_dirs(out_root: str | Path) -> dict:
    out = Path(out_root)
    dirs = {name: out / name for name in
            ("keyframes", "masks", "crops", "embeddings", "failure_cases")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dirs["root"] = out
    return dirs
