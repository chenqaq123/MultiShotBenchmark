# Multi-Shot Scene Consistency Benchmark — Evaluation Suite

Implementation of `pipeline_plan.md` (primary) / `proposal.md` (secondary): a
deterministic, auditable pipeline that scores multi-shot generated videos along
three tracks — prompt-specified entity consistency, model-emergent
self-consistency, and same-view background consistency — plus a grouping
quality self-check.

## Layout

```
src/                       evaluation suite (see pipeline_plan.md §22)
configs/                   episode configs (video + shots + entities + view labels)
outputs/<episode_id>/      keyframes/ masks/ crops/ embeddings/
                           shots.json keyframes.json proposals.json
                           entity_tracks.json same_view_groups.json metrics.json
                           view_confusion_matrix.png (when view labels given)
```

## Models

| Stage | Model |
|---|---|
| open-world / prompt grounding | GroundingDINO-tiny (transformers) |
| mask proposals | SAM ViT-B (transformers) |
| object / background features | DINOv2-base |
| depth layout | Depth-Anything-V2-Small |
| edge layout | OpenCV Canny |
| face detection + identity | InsightFace buffalo_l (SCRFD + ArcFace) |

Same-view score uses the no-geometry variant of pipeline_plan §15.1:
`0.60 * S_dino_bg + 0.25 * S_depth + 0.15 * S_edge`, p5/p95-normalized
within the episode, mutual-kNN graph + connected components.

## Run

```bash
# environment: venv over the latentGuard conda env (torch/cu128, transformers)
# /home/chenguanxu/venvs/msbench  (+ insightface)

export HF_HOME=/home/chenguanxu/hf_cache_msbench
export HF_HUB_CACHE=$HF_HOME/hub

/home/chenguanxu/venvs/msbench/bin/python -m src.run_pipeline \
    --episode configs/episode_blooper_01.json \
    --output outputs/blooper_01
```

Episode config fields: `video` (required), `entities` (optional structured
prompt-specified list, pipeline_plan §5.3; without it only the model-emergent
and background tracks are scored), `view_labels` (optional ground-truth view
labels per detected shot, enabling the view-grouping quality self-check).

Shot boundaries are always detected algorithmically from the video itself
(`src/detect_shots.py`: dual-signal HSV-histogram + downscaled-pixel
difference, robust MAD normalization, cross-gap verification that rejects
flash frames and chunked-generation seams). A config `shots` field is ignored.

All thresholds live in `src/common.py::DEFAULTS` and are echoed into
`metrics.json["config"]`. Metrics report coverage alongside similarity scores;
dimensions with no comparable evidence come out as `null` with coverage 0 —
no pass/fail verdicts (the `*_pass_rate` stats are parameterized by the
documented thresholds, per pipeline_plan §9.5/§10.5).
