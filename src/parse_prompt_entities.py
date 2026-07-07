"""Step 2: prompt-specified entity parsing.

The benchmark input is a structured entity list (pipeline_plan §5.3), provided
in the episode config either inline or via a separate JSON file. When absent,
the episode is evaluated on the model-emergent and background tracks only.
"""
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = {"id", "description"}


def parse_prompt_entities(episode_cfg: dict) -> dict:
    entities = episode_cfg.get("entities")
    if isinstance(entities, str):
        entities = json.loads(Path(entities).read_text(encoding="utf-8"))
    entities = entities or {}
    parsed = {"characters": [], "objects": [], "backgrounds": []}
    for kind in parsed:
        for ent in entities.get(kind, []):
            missing = REQUIRED_FIELDS - set(ent)
            if missing:
                raise ValueError(f"entity {ent} missing fields {missing}")
            parsed[kind].append({
                "id": ent["id"],
                "description": ent["description"].strip().lower(),
                "scheduled_shots": sorted(ent.get("scheduled_shots", [])),
            })
    return parsed
