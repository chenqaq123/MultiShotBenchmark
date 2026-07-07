"""Step 3: open-world proposal extraction with GroundingDINO + SAM masks.

Text queries combine a generic open-world vocabulary (for model-emergent
discovery) with the prompt-specified entity descriptions (for grounding).
Each proposal remembers which query fired so association (Step 5) can tell
prompt-driven detections from open-world ones.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import DEFAULTS, Keyframe, Proposal, box_iou, laplacian_blur
from .models import get_device, get_grounding_dino, get_sam

PERSON_LABELS = {"person", "man", "woman", "boy", "girl", "child", "face"}
BG_REGION_LABELS = {"picture frame", "window", "door", "curtain", "wall", "blinds"}


def _entity_queries(prompt_entities: dict) -> list[tuple[str, str]]:
    """[(description, entity_id)] — each grounded in its own forward pass so
    detections map to the entity directly (GroundingDINO returns token-level
    labels, not full query phrases, so label parsing cannot be trusted)."""
    return [(ent["description"], ent["id"])
            for kind in ("characters", "objects")
            for ent in prompt_entities.get(kind, [])]


def _classify(label: str, matched_entity: str | None, prompt_entities: dict) -> str:
    if matched_entity:
        if any(e["id"] == matched_entity for e in prompt_entities["characters"]):
            return "person"
        return "object"
    l = label.lower()
    if any(p in l.split() for p in PERSON_LABELS):
        return "person"
    if l in BG_REGION_LABELS:
        return "background-region"
    return "object"


def extract_proposals(keyframes: list[Keyframe], prompt_entities: dict,
                      crops_dir: str | Path, masks_dir: str | Path,
                      cfg: dict = DEFAULTS) -> list[Proposal]:
    import torch

    crops_dir, masks_dir = Path(crops_dir), Path(masks_dir)
    proc, model = get_grounding_dino()
    sam_proc, sam_model = get_sam()
    device = get_device()

    entity_queries = _entity_queries(prompt_entities)
    generic_text = ". ".join(cfg["open_world_vocab"]) + "."

    def _detect(rgb, text, size):
        inputs = proc(images=rgb, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=cfg["det_box_threshold"],
            text_threshold=cfg["det_text_threshold"],
            target_sizes=[size])[0]

    proposals: list[Proposal] = []
    for kf in keyframes:
        bgr = cv2.imread(kf.path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # prompt-entity passes first (their boxes win NMS over generic ones),
        # then the generic open-world pass
        raw: list[tuple[np.ndarray, float, str, str | None]] = []
        for desc, eid in entity_queries:
            res = _detect(rgb, desc + ".", (h, w))
            raw.extend((b, float(s), desc, eid) for b, s in
                       zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()))
        raw.sort(key=lambda r: -r[1])
        res = _detect(rgb, generic_text, (h, w))
        raw.extend(sorted(
            ((b, float(s), l, None) for b, s, l in
             zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                 res["text_labels"])),
            key=lambda r: -r[1]))

        kept: list[tuple[np.ndarray, float, str, str | None]] = []
        for box, score, label, eid in raw:
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
            area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            if not (cfg["prop_min_area_ratio"] <= area_ratio <= cfg["prop_max_area_ratio"]):
                continue
            if any(box_iou(box, kb) > cfg["prop_nms_iou"] for kb, _, _, _ in kept):
                continue  # duplicate across queries
            kept.append((box, float(score), label, eid))

        # blur filter on crops + build proposals
        kf_props: list[Proposal] = []
        for box, score, label, eid in kept:
            x1, y1, x2, y2 = (int(v) for v in box)
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if laplacian_blur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < cfg["prop_min_blur"]:
                continue
            kind = _classify(label, eid, prompt_entities)
            pid = f"{kf.kf_id}_p{len(kf_props):02d}"
            crop_path = crops_dir / f"{pid}.png"
            cv2.imwrite(str(crop_path), crop)
            kf_props.append(Proposal(
                prop_id=pid, kf_id=kf.kf_id, shot_index=kf.shot_index,
                box=[float(x1), float(y1), float(x2), float(y2)],
                score=round(float(score), 4), label=label, kind=kind,
                area_ratio=round(float((x2 - x1) * (y2 - y1) / (w * h)), 4),
                crop_path=str(crop_path), from_entity_query=eid))

        # SAM masks for all proposals of this keyframe (box prompts)
        if kf_props:
            boxes = [[p.box for p in kf_props]]
            sam_inputs = sam_proc(rgb, input_boxes=boxes, return_tensors="pt").to(device)
            with torch.no_grad():
                sam_out = sam_model(**sam_inputs, multimask_output=False)
            masks = sam_proc.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu())[0]  # (n, 1, H, W) bool
            mask_arr = masks[:, 0].numpy().astype(bool)
            np.savez_compressed(masks_dir / f"{kf.kf_id}_proposals.npz",
                                prop_ids=np.array([p.prop_id for p in kf_props]),
                                masks=np.packbits(mask_arr, axis=None),
                                shape=np.array(mask_arr.shape))
        proposals.extend(kf_props)
    return proposals


def load_proposal_masks(masks_dir: str | Path, kf_id: str) -> dict[str, np.ndarray]:
    path = Path(masks_dir) / f"{kf_id}_proposals.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    shape = tuple(data["shape"])
    masks = np.unpackbits(data["masks"], axis=None)[: int(np.prod(shape))]
    masks = masks.reshape(shape).astype(bool)
    return {pid: masks[i] for i, pid in enumerate(data["prop_ids"])}
