"""Shot boundary detection — the pipeline's sole source of shot structure.

Shot segmentation never relies on prompt / config shot lists; it is computed
from the video itself:

1. Two frame-difference signals per frame transition t -> t+1:
   - d_hist: 1 - correlation of HSV histograms (palette change)
   - d_pix:  mean absolute difference of downscaled gray frames (layout change)
2. Each signal is robustly normalized (median + MAD) over the episode; a
   candidate cut is a local maximum whose normalized score exceeds
   `shot_adaptive_k` and whose raw magnitude clears an absolute floor.
3. Cross-gap verification: a real hard cut keeps the content different a few
   frames past the boundary, while flash artifacts and chunked-generation
   seams (common in generated video) recover — so frames t-gap and t+1+gap
   must still differ by a comparable amount for the candidate to survive.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import DEFAULTS, Shot


def _frame_signature(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    gray = cv2.cvtColor(cv2.resize(frame, (48, 27)), cv2.COLOR_BGR2GRAY)
    return hist, gray.astype(np.float32) / 255.0


def _hist_diff(h1, h2) -> float:
    return 1.0 - cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def _pix_diff(g1, g2) -> float:
    return float(np.abs(g1 - g2).mean())


def detect_shots(video_path: str, cfg: dict = DEFAULTS) -> list[Shot]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_len = int(cfg["shot_min_len_sec"] * fps)

    hists, grays = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist, gray = _frame_signature(frame)
        hists.append(hist)
        grays.append(gray)
    cap.release()
    total = len(hists)
    if total == 0:
        raise RuntimeError(f"no frames decoded from: {video_path}")

    d_hist = np.array([_hist_diff(hists[t], hists[t + 1]) for t in range(total - 1)])
    d_pix = np.array([_pix_diff(grays[t], grays[t + 1]) for t in range(total - 1)])

    def robust_z(d: np.ndarray) -> np.ndarray:
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) + 1e-6
        return (d - med) / mad

    score = np.maximum(robust_z(d_hist), robust_z(d_pix))
    gap = int(cfg["shot_verify_gap"])

    cuts = [0]
    for t in range(len(score)):
        lo, hi = max(0, t - 2), min(len(score), t + 3)
        if score[t] < cfg["shot_adaptive_k"] or score[t] != score[lo:hi].max():
            continue
        # only channels whose step diff clears the absolute floor count as fired
        fired_hist = d_hist[t] >= cfg["shot_min_diff"]
        fired_pix = d_pix[t] >= cfg["shot_min_pix_diff"]
        if not (fired_hist or fired_pix):
            continue  # statistically unusual but visually negligible
        # cross-gap verification on the fired channel(s): a real cut keeps the
        # content changed past the boundary (cross diff comparable to the step
        # diff); flash frames and chunked-generation seams recover, so their
        # cross diff collapses back to the motion baseline
        a, b = max(0, t - gap), min(total - 1, t + 1 + gap)
        keeps = False
        if fired_hist and _hist_diff(hists[a], hists[b]) >= max(
                0.5 * d_hist[t], cfg["shot_min_diff"]):
            keeps = True
        if fired_pix and _pix_diff(grays[a], grays[b]) >= max(
                0.5 * d_pix[t], cfg["shot_min_pix_diff"]):
            keeps = True
        if not keeps:
            continue  # transient, not a cut
        if (t + 1) - cuts[-1] >= min_len:
            cuts.append(t + 1)  # cut between frame t and t+1
    cuts.append(total)

    shots = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s < max(1, min_len // 2):
            continue
        shots.append(Shot(index=len(shots) + 1, start_frame=s, end_frame=e,
                          start_sec=s / fps, end_sec=e / fps))
    return shots
