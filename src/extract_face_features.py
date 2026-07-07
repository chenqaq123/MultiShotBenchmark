"""Step 6 support: face detection + ArcFace embeddings (InsightFace buffalo_l).

Faces are detected on the full keyframe, then attached to the smallest person
proposal whose box contains the face center.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import Keyframe, Proposal
from .models import get_face_app


def _contains(box, x, y) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def extract_face_features(keyframes: list[Keyframe], proposals: list[Proposal]) -> int:
    """Sets face_embedding / face_det_score on person proposals. Returns #faces."""
    app = get_face_app()
    by_kf: dict[str, list[Proposal]] = {}
    for p in proposals:
        if p.kind == "person":
            by_kf.setdefault(p.kf_id, []).append(p)

    n_attached = 0
    for kf in keyframes:
        persons = by_kf.get(kf.kf_id, [])
        if not persons:
            continue
        bgr = cv2.imread(kf.path)
        faces = app.get(bgr)
        for face in faces:
            fx = (face.bbox[0] + face.bbox[2]) / 2
            fy = (face.bbox[1] + face.bbox[3]) / 2
            # attach to every containing person-kind proposal (person boxes and
            # face boxes both carry the identity evidence of this face)
            for target in (p for p in persons if _contains(p.box, fx, fy)):
                if target.face_det_score is None or face.det_score > target.face_det_score:
                    emb = face.normed_embedding.astype(np.float32)
                    target.face_embedding = emb.tolist()
                    target.face_det_score = float(face.det_score)
                    n_attached += 1
    return n_attached
