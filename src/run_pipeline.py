"""End-to-end orchestrator for the multi-shot consistency evaluation suite.

Usage:
    python -m src.run_pipeline --episode configs/episode.json --output outputs/<id>

Episode config:
    {
      "episode_id": "...",
      "video": "path/to/video.mp4",
      "entities": {"characters": [...], "objects": [...], "backgrounds": [...]},  # optional
      "view_labels": {"1": "view_a", "2": "view_b", ...}   # optional, per detected shot
    }

    Shot boundaries are always detected algorithmically from the video
    (src/detect_shots.py); a config "shots" field is ignored.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from .associate_entities import associate_entities
from .cluster_same_view import cluster_same_view
from .common import DEFAULTS, ensure_dirs, load_episode_config, save_json
from .compute_same_view_score import compute_same_view_scores
from .detect_shots import detect_shots
from .evaluate_metrics import evaluate_metrics
from .export_failure_cases import export_failure_cases
from .extract_background_features import extract_background_features
from .extract_face_features import extract_face_features
from .extract_keyframes import extract_keyframes
from .extract_object_features import extract_object_features
from .extract_open_world_proposals import extract_proposals
from .parse_prompt_entities import parse_prompt_entities
from .segment_foreground_background import segment_foreground_background


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_episode(episode_cfg_path: str, output_dir: str, device: str | None = None,
                cfg: dict = DEFAULTS) -> dict:
    if device:
        from . import models
        models.set_device(device)

    ep = load_episode_config(episode_cfg_path)
    dirs = ensure_dirs(output_dir)
    log(f"episode={ep.get('episode_id', '?')} video={ep['video']}")

    # Step 0/1: shots (always algorithmic — never taken from prompt/config) + keyframes
    if ep.get("shots"):
        log("note: config 'shots' is ignored; shot boundaries are always detected "
            "from the video itself")
    shots = detect_shots(ep["video"], cfg)
    log(f"shots detected: {len(shots)} -> "
        + ", ".join(f"{s.start_sec:.1f}-{s.end_sec:.1f}s" for s in shots))
    save_json([asdict(s) for s in shots], dirs["root"] / "shots.json")

    keyframes = extract_keyframes(ep["video"], shots, dirs["keyframes"], cfg)
    log(f"keyframes: {len(keyframes)}")

    # Step 2: prompt-specified entities
    entities = parse_prompt_entities(ep)
    n_ent = sum(len(v) for v in entities.values())
    log(f"prompt entities: {n_ent} "
        f"(chars={len(entities['characters'])}, objs={len(entities['objects'])}, "
        f"bgs={len(entities['backgrounds'])})")

    # Step 3: open-world proposals (+ SAM masks)
    proposals = extract_proposals(keyframes, entities, dirs["crops"], dirs["masks"], cfg)
    log(f"proposals: {len(proposals)} "
        f"(person={sum(p.kind == 'person' for p in proposals)}, "
        f"object={sum(p.kind == 'object' for p in proposals)}, "
        f"bg-region={sum(p.kind == 'background-region' for p in proposals)})")

    # Step 4: foreground / background separation
    segment_foreground_background(keyframes, proposals, dirs["masks"])
    save_json([asdict(k) for k in keyframes], dirs["root"] / "keyframes.json")

    # Steps 6/7 features
    n_faces = extract_face_features(keyframes, proposals)
    log(f"faces attached to person proposals: {n_faces}")
    feats = extract_object_features(proposals, dirs["embeddings"])
    log(f"DINOv2 proposal embeddings: {len(feats['dino'])}")

    # Step 5: association (prompt-specified + model-emergent tracks)
    tracks = associate_entities(proposals, entities, feats, cfg)
    log(f"emergent tracks: chars={len(tracks['emergent']['characters'])}, "
        f"objs={len(tracks['emergent']['objects'])}")

    # Step 8: background features + same-view grouping
    bg_feats = extract_background_features(keyframes, dirs["masks"], dirs["embeddings"], cfg)
    scores = compute_same_view_scores(keyframes, bg_feats, cfg)
    same_view = cluster_same_view(scores, keyframes, cfg)
    log(f"same-view groups: {len(same_view['groups'])} "
        f"(multi-frame={sum(g['size'] > 1 for g in same_view['groups'])}, "
        f"excluded kfs={len(same_view['excluded_kf_ids'])})")

    # Steps 9/10: metrics
    metrics = evaluate_metrics(tracks, proposals, keyframes, feats,
                               same_view, scores, ep.get("view_labels"), cfg)
    metrics["episode_id"] = ep.get("episode_id")
    metrics["video"] = ep["video"]

    n_cases = export_failure_cases(metrics, proposals, dirs["failure_cases"], cfg)
    log(f"failure-case exhibits exported: {n_cases}")

    # persist evidence
    save_json([asdict(p) for p in proposals], dirs["root"] / "proposals.json")
    tracks_out = {**tracks, "emergent": {**tracks["emergent"]}}
    for key in ("characters", "characters_no_face", "objects"):
        tracks_out["emergent"][key] = [
            {k: v for k, v in t.items() if k != "centroid"}
            for t in tracks["emergent"][key]]
    save_json(tracks_out, dirs["root"] / "entity_tracks.json")
    save_json(same_view, dirs["root"] / "same_view_groups.json")
    save_json(metrics, dirs["root"] / "metrics.json")
    _plot_confusion(metrics, dirs["root"])
    log(f"done -> {dirs['root'] / 'metrics.json'}")
    return metrics


def _plot_confusion(metrics: dict, out_root: Path) -> None:
    q = metrics.get("view_grouping_quality", {})
    if not q.get("available"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cm = q["view_confusion_matrix"]
    rows = sorted(cm)
    cols = sorted({c for r in cm.values() for c in r})
    mat = np.array([[cm[r].get(c, 0) for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(cols), 1.2 + 0.6 * len(rows)))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, mat[i, j], ha="center", va="center")
    ax.set_xlabel("ground-truth view")
    ax.set_ylabel("predicted group")
    fig.tight_layout()
    fig.savefig(out_root / "view_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", required=True, help="episode config JSON")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 / cpu (default: auto)")
    args = ap.parse_args()
    run_episode(args.episode, args.output, args.device)


if __name__ == "__main__":
    main()
