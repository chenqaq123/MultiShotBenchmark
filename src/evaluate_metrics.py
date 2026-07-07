"""Steps 9–10: final metric report (four groups, no single blended score).

1. prompt_specified   – presence + cross-shot identity/appearance stability
2. model_emergent     – count / recurrence / self-consistency / fragmentation
3. background_same_view – intra-group background & layout consistency
4. view_grouping_quality – only when ground-truth view labels are provided

Dimensions without comparable evidence report coverage=0 and null scores
(no pass/fail verdicts; thresholds only parameterize the *_pass_rate stats
required by pipeline_plan §9.5/§10.5).
"""
from __future__ import annotations

import numpy as np

from .common import DEFAULTS, Keyframe, Proposal, cosine


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _pair_stats(items: list[tuple], embs: list[np.ndarray], threshold: float,
                sim_fn=None) -> dict:
    """items[i] = (shot, kf_id, prop_id) aligned with embs[i].

    Pairwise similarity uses sim_fn(i, j) when given (e.g. blended object
    similarity), cosine of embs otherwise. Centroid similarity (pipeline_plan
    §9.4) is always computed from embs: cos(emb_i, mean of normalized embs).
    """
    empty = {"n_pairs": 0, "mean": None, "min": None, "std": None,
             "pass_rate": None, "centroid_mean": None, "centroid_min": None,
             "pairs": [], "lowest_pair": None}
    if len(embs) < 2:
        return empty
    pairs = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = sim_fn(i, j) if sim_fn else cosine(embs[i], embs[j])
            pairs.append({"a": items[i], "b": items[j],
                          "similarity": round(float(sim), 4)})
    sims = np.array([p["similarity"] for p in pairs])
    lowest = min(pairs, key=lambda p: p["similarity"])
    normed = [np.asarray(e, np.float32) / (np.linalg.norm(e) + 1e-8) for e in embs]
    centroid = np.mean(normed, axis=0)
    csims = np.array([cosine(e, centroid) for e in normed])
    return {"n_pairs": len(pairs),
            "mean": round(float(sims.mean()), 4),
            "min": round(float(sims.min()), 4),
            "std": round(float(sims.std()), 4),
            "pass_rate": round(float((sims >= threshold).mean()), 4),
            "centroid_mean": round(float(csims.mean()), 4),
            "centroid_min": round(float(csims.min()), 4),
            "pairs": pairs, "lowest_pair": lowest}


def _agg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


# --------------------------------------------------------------------------
# group 1: prompt-specified entity consistency
# --------------------------------------------------------------------------
def _prompt_character_metrics(records: list[dict], prop_by_id: dict[str, Proposal],
                              cfg: dict) -> dict:
    details, presence, det_rates = [], [], []
    for rec in records:
        scheduled = [a for a in rec["appearances"] if a.get("scheduled", True)]
        n_sched = len(scheduled) + len(rec["missing_shots"])
        pres = len(scheduled) / n_sched if n_sched else None
        apps = rec["appearances"]
        faced = [(a, prop_by_id[a["prop_id"]]) for a in apps
                 if prop_by_id[a["prop_id"]].face_embedding]
        det = len(faced) / len(apps) if apps else None
        stats = _pair_stats(
            [(a["shot"], a["kf_id"], a["prop_id"]) for a, _ in faced],
            [np.asarray(p.face_embedding, np.float32) for _, p in faced],
            cfg["face_match_threshold"])
        presence.append(pres)
        det_rates.append(det)
        details.append({"entity_id": rec["id"], "description": rec["description"],
                        "scheduled_shots": rec["scheduled_shots"],
                        "missing_shots": rec["missing_shots"],
                        "presence_rate": pres, "face_detection_rate": det,
                        "face_similarity": stats})
    return {
        "prompt_character_presence_rate": _agg(presence),
        "prompt_face_detection_rate": _agg(det_rates),
        "prompt_face_mean_similarity": _agg([d["face_similarity"]["mean"] for d in details]),
        "prompt_face_min_similarity": _agg([d["face_similarity"]["min"] for d in details]),
        "prompt_face_similarity_std": _agg([d["face_similarity"]["std"] for d in details]),
        "prompt_face_centroid_similarity": _agg([d["face_similarity"]["centroid_mean"] for d in details]),
        "prompt_identity_pass_rate": _agg([d["face_similarity"]["pass_rate"] for d in details]),
        "coverage": {"n_entities": len(records),
                     "n_comparable_pairs": sum(d["face_similarity"]["n_pairs"] for d in details)},
        "per_entity": details,
    }


def _prompt_object_metrics(records: list[dict], prop_by_id: dict[str, Proposal],
                           feats: dict, cfg: dict) -> dict:
    from .extract_object_features import object_pair_similarity
    details, presence = [], []
    for rec in records:
        scheduled = [a for a in rec["appearances"] if a.get("scheduled", True)]
        n_sched = len(scheduled) + len(rec["missing_shots"])
        pres = len(scheduled) / n_sched if n_sched else None
        apps = [(a, prop_by_id[a["prop_id"]]) for a in rec["appearances"]
                if prop_by_id[a["prop_id"]].prop_id in feats["dino"]]
        props = [p for _, p in apps]
        stats = _pair_stats(
            [(a["shot"], a["kf_id"], a["prop_id"]) for a, _ in apps],
            [feats["dino"][p.prop_id] for p in props],
            cfg["object_match_threshold"],
            sim_fn=lambda i, j: object_pair_similarity(props[i], props[j], feats, cfg))
        presence.append(pres)
        details.append({"entity_id": rec["id"], "description": rec["description"],
                        "scheduled_shots": rec["scheduled_shots"],
                        "missing_shots": rec["missing_shots"],
                        "presence_rate": pres, "object_similarity": stats})
    return {
        "prompt_object_presence_rate": _agg(presence),
        "prompt_object_mean_similarity": _agg([d["object_similarity"]["mean"] for d in details]),
        "prompt_object_min_similarity": _agg([d["object_similarity"]["min"] for d in details]),
        "prompt_object_similarity_std": _agg([d["object_similarity"]["std"] for d in details]),
        "prompt_object_centroid_similarity": _agg([d["object_similarity"]["centroid_mean"] for d in details]),
        "prompt_object_pass_rate": _agg([d["object_similarity"]["pass_rate"] for d in details]),
        "coverage": {"n_entities": len(records),
                     "n_comparable_pairs": sum(d["object_similarity"]["n_pairs"] for d in details)},
        "per_entity": details,
    }


# --------------------------------------------------------------------------
# group 2: model-emergent self-consistency
# --------------------------------------------------------------------------
def _emergent_metrics(emergent: dict, prop_by_id: dict[str, Proposal],
                      feats: dict, cfg: dict) -> dict:
    from .extract_object_features import object_pair_similarity

    def track_stats(tracks, get_emb, threshold, blended=False):
        details = []
        for t in tracks:
            items, embs, props = [], [], []
            for pid in t["prop_ids"]:
                e = get_emb(prop_by_id[pid])
                if e is not None:
                    items.append((prop_by_id[pid].shot_index, prop_by_id[pid].kf_id, pid))
                    embs.append(e)
                    props.append(prop_by_id[pid])
            sim_fn = (lambda i, j: object_pair_similarity(props[i], props[j], feats, cfg)) \
                if blended else None
            stats = _pair_stats(items, embs, threshold, sim_fn=sim_fn)
            details.append({"track_id": t["track_id"], "shots": t["shots"],
                            "labels": t["labels"], "n_members": len(t["prop_ids"]),
                            "similarity": stats})
        return details

    char_details = track_stats(
        emergent["characters"],
        lambda p: np.asarray(p.face_embedding, np.float32) if p.face_embedding else None,
        cfg["face_match_threshold"])
    obj_details = track_stats(
        emergent["objects"],
        lambda p: feats["dino"].get(p.prop_id),
        cfg["object_match_threshold"], blended=True)

    n_char_clu = emergent["character_cluster_total"]
    n_obj_clu = emergent["object_cluster_total"]
    return {
        "emergent_character_count": len(emergent["characters"]),
        "emergent_character_recurrence_rate":
            round(len(emergent["characters"]) / n_char_clu, 4) if n_char_clu else None,
        "emergent_face_mean_similarity": _agg([d["similarity"]["mean"] for d in char_details]),
        "emergent_face_min_similarity": _agg([d["similarity"]["min"] for d in char_details]),
        "emergent_face_similarity_std": _agg([d["similarity"]["std"] for d in char_details]),
        "emergent_face_centroid_similarity": _agg([d["similarity"]["centroid_mean"] for d in char_details]),
        "emergent_identity_fragmentation_rate":
            round(emergent["character_fragmentation_rate"], 4),
        "emergent_object_count": len(emergent["objects"]),
        "emergent_object_recurrence_rate":
            round(len(emergent["objects"]) / n_obj_clu, 4) if n_obj_clu else None,
        "emergent_object_mean_similarity": _agg([d["similarity"]["mean"] for d in obj_details]),
        "emergent_object_min_similarity": _agg([d["similarity"]["min"] for d in obj_details]),
        "emergent_object_similarity_std": _agg([d["similarity"]["std"] for d in obj_details]),
        "emergent_object_centroid_similarity": _agg([d["similarity"]["centroid_mean"] for d in obj_details]),
        "emergent_object_fragmentation_rate":
            round(emergent["object_fragmentation_rate"], 4),
        "coverage": {"character_cluster_total": n_char_clu,
                     "object_cluster_total": n_obj_clu,
                     "characters_no_face_tracks": len(emergent["characters_no_face"])},
        "per_track": {"characters": char_details, "objects": obj_details},
    }


# --------------------------------------------------------------------------
# group 3: background same-view consistency
# --------------------------------------------------------------------------
def _background_metrics(groups: list[dict], scores: dict) -> dict:
    idx = {k: i for i, k in enumerate(scores["kf_ids"])}
    per_group, weights = [], []
    for g in groups:
        members = [idx[k] for k in g["kf_ids"]]
        n = len(members)
        n_pairs = n * (n - 1) // 2
        if n_pairs == 0:
            continue
        def mean_of(mat):
            vals = [mat[i, j] for a, i in enumerate(members)
                    for j in members[a + 1:]]
            return round(float(np.mean(vals)), 4)
        per_group.append({
            "group_id": g["group_id"], "size": n, "n_pairs": n_pairs,
            "medoid": g["medoid"], "kf_ids": g["kf_ids"],
            "intra_group_bg_similarity": mean_of(scores["raw"]["dino_bg"]),
            "intra_group_depth_similarity": mean_of(scores["raw"]["depth"]),
            "intra_group_edge_similarity": mean_of(scores["raw"]["edge"]),
            "intra_group_same_view_score": mean_of(scores["combined"]),
        })
        weights.append(n_pairs)
    multi = [g for g in groups if g["size"] > 1]
    w = np.array(weights, dtype=float)
    episode = (round(float(np.average(
        [g["intra_group_same_view_score"] for g in per_group], weights=w)), 4)
        if per_group else None)
    return {
        "same_view_group_count": len(groups),
        "same_view_multi_frame_group_count": len(multi),
        "average_same_view_group_size": round(float(np.mean([g["size"] for g in groups])), 4)
            if groups else None,
        "intra_group_bg_similarity": _agg([g["intra_group_bg_similarity"] for g in per_group]),
        "intra_group_depth_similarity": _agg([g["intra_group_depth_similarity"] for g in per_group]),
        "intra_group_edge_similarity": _agg([g["intra_group_edge_similarity"] for g in per_group]),
        "episode_same_view_consistency": episode,
        "coverage": {"n_grouped_keyframes": len(scores["kf_ids"]),
                     "n_excluded_keyframes": len(scores["excluded_kf_ids"]),
                     "excluded_kf_ids": scores["excluded_kf_ids"],
                     "n_comparable_pairs": int(sum(weights))},
        "per_group": per_group,
    }


# --------------------------------------------------------------------------
# group 4: view grouping quality (needs ground-truth view labels per shot)
# --------------------------------------------------------------------------
def _grouping_quality(groups: list[dict], keyframes: list[Keyframe],
                      view_labels: dict | None) -> dict:
    if not view_labels:
        return {"available": False,
                "note": "no ground-truth view labels provided; grouping self-check skipped"}
    shot_of = {kf.kf_id: kf.shot_index for kf in keyframes}
    group_of = {k: g["group_id"] for g in groups for k in g["kf_ids"]}
    kfs = [k for k in group_of if str(shot_of[k]) in view_labels]
    tp = fp = fn = tn = 0
    for i in range(len(kfs)):
        for j in range(i + 1, len(kfs)):
            same_pred = group_of[kfs[i]] == group_of[kfs[j]]
            same_gt = view_labels[str(shot_of[kfs[i]])] == view_labels[str(shot_of[kfs[j]])]
            tp += same_pred and same_gt
            fp += same_pred and not same_gt
            fn += (not same_pred) and same_gt
            tn += (not same_pred) and (not same_gt)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    confusion: dict[str, dict[str, int]] = {}
    for k in kfs:
        confusion.setdefault(group_of[k], {}).setdefault(
            view_labels[str(shot_of[k])], 0)
        confusion[group_of[k]][view_labels[str(shot_of[k])]] += 1
    return {"available": True,
            "pairwise_precision": round(prec, 4) if prec is not None else None,
            "pairwise_recall": round(rec, 4) if rec is not None else None,
            "pairwise_f1": round(f1, 4) if f1 is not None else None,
            "over_merge_rate": round(fp / (tp + fp), 4) if tp + fp else None,
            "over_split_rate": round(fn / (tp + fn), 4) if tp + fn else None,
            "pair_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "view_confusion_matrix": confusion}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def evaluate_metrics(tracks: dict, proposals: list[Proposal], keyframes: list[Keyframe],
                     feats: dict, same_view: dict, scores: dict,
                     view_labels: dict | None = None, cfg: dict = DEFAULTS) -> dict:
    prop_by_id = {p.prop_id: p for p in proposals}
    groups = same_view["groups"]
    char_m = _prompt_character_metrics(tracks["prompt"]["characters"], prop_by_id, cfg)
    obj_m = _prompt_object_metrics(tracks["prompt"]["objects"], prop_by_id, feats, cfg)
    return {
        "prompt_specified": {
            **{k: v for k, v in char_m.items() if k not in ("coverage", "per_entity")},
            **{k: v for k, v in obj_m.items() if k not in ("coverage", "per_entity")},
            "coverage": {"characters": char_m["coverage"], "objects": obj_m["coverage"]},
            "per_entity": {"characters": char_m["per_entity"],
                           "objects": obj_m["per_entity"]},
        },
        "model_emergent": _emergent_metrics(tracks["emergent"], prop_by_id, feats, cfg),
        "background_same_view": _background_metrics(groups, scores),
        "view_grouping_quality": _grouping_quality(groups, keyframes, view_labels),
        "config": {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, list, dict))},
    }
