"""Lazy singletons for all pretrained vision models used by the suite."""
from __future__ import annotations

import numpy as np

from .common import DEFAULTS, pick_device

_cache: dict = {}


def get_device() -> str:
    if "device" not in _cache:
        _cache["device"] = pick_device()
    return _cache["device"]


def set_device(device: str) -> None:
    _cache["device"] = device


def get_grounding_dino():
    if "gdino" not in _cache:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        name = DEFAULTS["gdino_model"]
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(name)
        model.to(get_device()).eval()
        _cache["gdino"] = (proc, model)
    return _cache["gdino"]


def get_sam():
    if "sam" not in _cache:
        from transformers import SamModel, SamProcessor
        name = DEFAULTS["sam_model"]
        proc = SamProcessor.from_pretrained(name)
        model = SamModel.from_pretrained(name)
        model.to(get_device()).eval()
        _cache["sam"] = (proc, model)
    return _cache["sam"]


def get_dinov2():
    if "dinov2" not in _cache:
        from transformers import AutoImageProcessor, AutoModel
        name = DEFAULTS["dino_model"]
        proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.to(get_device()).eval()
        _cache["dinov2"] = (proc, model)
    return _cache["dinov2"]


def get_depth():
    if "depth" not in _cache:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = DEFAULTS["depth_model"]
        proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModelForDepthEstimation.from_pretrained(name)
        model.to(get_device()).eval()
        _cache["depth"] = (proc, model)
    return _cache["depth"]


def get_face_app():
    if "face" not in _cache:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection", "recognition"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _cache["face"] = app
    return _cache["face"]


def dinov2_embed_images(images_rgb: list[np.ndarray]) -> np.ndarray:
    """CLS embeddings (L2-normalized) for a list of RGB uint8 images."""
    import torch
    proc, model = get_dinov2()
    embs = []
    with torch.no_grad():
        for i in range(0, len(images_rgb), 16):
            batch = images_rgb[i:i + 16]
            inputs = proc(images=batch, return_tensors="pt").to(model.device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            embs.append(cls.cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), np.float32)
