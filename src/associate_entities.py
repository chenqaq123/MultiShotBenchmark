"""Step 5: entity association.

A. Proposals fired by a prompt-entity description query are assigned to that
   entity (best score per scheduled shot; unmatched scheduled shots => missing).
B. Remaining person/object proposals form the model-emergent pool and are
   clustered across shots (faces by ArcFace embedding, the rest by DINOv2).
"""
from __future__ import annotations

import numpy as np

from .common import DEFAULTS, Proposal, box_iou, cosine


def _cluster(embs: np.ndarray, threshold_sim: float) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance."""
    from sklearn.cluster import AgglomerativeClustering
    if len(embs) == 0:
        return np.array([], dtype=int)
    if len(embs) == 1:
        return np.array([0])
    clu = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                  linkage="average",
                                  distance_threshold=1.0 - threshold_sim)
    return clu.fit_predict(embs)


def _assign_prompt_entities(entities: list[dict], proposals: list[Proposal],
                            shot_of_kf: dict[str, int]) -> list[dict]:
    records = []
    for ent in entities:
        cands = [p for p in proposals if p.from_entity_query == ent["id"]]
        by_shot: dict[int, list[Proposal]] = {}
        for p in cands:
            by_shot.setdefault(p.shot_index, []).append(p)
        appearances, missing = [], []
        shots = ent["scheduled_shots"] or sorted(by_shot)
        for shot in shots:
            best = max(by_shot.get(shot, []), key=lambda p: p.score, default=None)
            if best is None:
                missing.append(shot)
            else:
                best.assigned_entity = ent["id"]
                appearances.append({"shot": shot, "kf_id": best.kf_id,
                                    "prop_id": best.prop_id, "score": best.score})
        # extra (non-scheduled) appearances, best per shot
        for shot in sorted(set(by_shot) - set(shots)):
            best = max(by_shot[shot], key=lambda p: p.score)
            best.assigned_entity = ent["id"]
            appearances.append({"shot": shot, "kf_id": best.kf_id,
                                "prop_id": best.prop_id, "score": best.score,
                                "scheduled": False})
        records.append({**ent, "appearances": appearances, "missing_shots": missing})
    return records


def _emergent_pool(proposals: list[Proposal]) -> list[Proposal]:
    # prompt evidence per keyframe: assigned proposals AND every proposal fired
    # by an entity-description query (best-per-shot assignment leaves the same
    # entity's detections in sibling keyframes unassigned)
    assigned_by_kf: dict[str, list[Proposal]] = {}
    for p in proposals:
        if p.assigned_entity or p.from_entity_query:
            assigned_by_kf.setdefault(p.kf_id, []).append(p)
    pool = []
    for p in proposals:
        if p.kind not in ("person", "object") or p.assigned_entity:
            continue
        if p.from_entity_query:
            continue  # duplicate grounding evidence of a prompt entity
        # same instance as (or part of) an assigned prompt proposal => not emergent;
        # containment, not IoU: a face box inside an assigned person box must match
        if any(_overlap_over_smaller(p.box, a.box) > 0.7
               for a in assigned_by_kf.get(p.kf_id, [])):
            continue
        pool.append(p)
    return pool


def _overlap_over_smaller(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    smaller = min(a1, a2)
    return inter / smaller if smaller > 0 else 0.0


def _make_tracks(pool: list[Proposal], embs: dict[str, np.ndarray],
                 threshold_sim: float, prefix: str) -> tuple[list[dict], int]:
    """Cluster pool by embedding; returns (recurring_tracks, n_clusters_total)."""
    items = [p for p in pool if p.prop_id in embs]
    if not items:
        return [], 0
    X = np.stack([embs[p.prop_id] for p in items])
    labels = _cluster(X, threshold_sim)
    tracks, n_total = [], len(set(labels))
    for lbl in sorted(set(labels)):
        members = [p for p, l in zip(items, labels) if l == lbl]
        # one member per keyframe (a person box and its face box share the same
        # identity evidence — keep the larger box)
        by_kf: dict[str, Proposal] = {}
        for p in members:
            area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
            cur = by_kf.get(p.kf_id)
            if cur is None or area > (cur.box[2] - cur.box[0]) * (cur.box[3] - cur.box[1]):
                by_kf[p.kf_id] = p
        members = list(by_kf.values())
        shots = sorted({p.shot_index for p in members})
        if len(shots) < 2:
            continue  # emergent track must recur across >=2 shots
        tid = f"{prefix}_{len(tracks):02d}"
        for p in members:
            p.track_id = tid
        centroid = X[labels == lbl].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        tracks.append({"track_id": tid, "shots": shots,
                       "prop_ids": [p.prop_id for p in members],
                       "labels": sorted({p.label for p in members}),
                       "centroid": centroid})
    return tracks, n_total


def _fragmentation_rate(tracks: list[dict], threshold_sim: float, margin: float) -> float:
    if len(tracks) < 2:
        return 0.0
    frag = set()
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if cosine(tracks[i]["centroid"], tracks[j]["centroid"]) > threshold_sim - margin:
                frag.update((i, j))
    return len(frag) / len(tracks)


def associate_entities(proposals: list[Proposal], prompt_entities: dict,
                       dino_embs: np.ndarray, cfg: dict = DEFAULTS) -> dict:
    shot_of_kf = {p.kf_id: p.shot_index for p in proposals}

    prompt_chars = _assign_prompt_entities(prompt_entities["characters"],
                                           [p for p in proposals if p.kind == "person"],
                                           shot_of_kf)
    prompt_objs = _assign_prompt_entities(prompt_entities["objects"],
                                          [p for p in proposals if p.kind == "object"],
                                          shot_of_kf)

    pool = _emergent_pool(proposals)
    face_embs = {p.prop_id: np.asarray(p.face_embedding, np.float32)
                 for p in pool if p.kind == "person" and p.face_embedding}
    dino_by_pid = {p.prop_id: dino_embs[p.dino_index] for p in pool
                   if p.dino_index is not None}
    faceless = {pid: e for pid, e in dino_by_pid.items()
                if pid not in face_embs and
                next(p for p in pool if p.prop_id == pid).kind == "person"}
    obj_embs = {pid: e for pid, e in dino_by_pid.items()
                if next(p for p in pool if p.prop_id == pid).kind == "object"}

    char_tracks, n_char_clu = _make_tracks(
        [p for p in pool if p.kind == "person" and p.prop_id in face_embs],
        face_embs, cfg["face_cluster_threshold"], "emergent_char")
    charless_tracks, n_charless_clu = _make_tracks(
        [p for p in pool if p.prop_id in faceless],
        faceless, cfg["object_cluster_threshold"], "emergent_person_noface")
    obj_tracks, n_obj_clu = _make_tracks(
        [p for p in pool if p.prop_id in obj_embs],
        obj_embs, cfg["object_cluster_threshold"], "emergent_obj")

    return {
        "prompt": {"characters": prompt_chars, "objects": prompt_objs,
                   "backgrounds": prompt_entities["backgrounds"]},
        "emergent": {
            "characters": char_tracks,
            "characters_no_face": charless_tracks,
            "objects": obj_tracks,
            "character_cluster_total": n_char_clu,
            "character_no_face_cluster_total": n_charless_clu,
            "object_cluster_total": n_obj_clu,
            "character_fragmentation_rate": _fragmentation_rate(
                char_tracks, cfg["face_cluster_threshold"], cfg["frag_margin"]),
            "object_fragmentation_rate": _fragmentation_rate(
                obj_tracks, cfg["object_cluster_threshold"], cfg["frag_margin"]),
        },
    }
