"""Step 1: shot-level keyframe extraction with low-quality frame filtering."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import DEFAULTS, Keyframe, Shot, laplacian_blur


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def _quality(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return laplacian_blur(gray), float(gray.mean())


def extract_keyframes(video_path: str, shots: list[Shot], out_dir: str | Path,
                      cfg: dict = DEFAULTS) -> list[Keyframe]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    keyframes: list[Keyframe] = []
    for shot in shots:
        n = shot.end_frame - shot.start_frame
        picked_times: list[float] = []
        positions = cfg["kf_positions"] if n > 3 * cfg["kf_search_window"] else [0.5]
        for pos in positions:
            center = shot.start_frame + int(pos * (n - 1))
            best, best_blur = None, -1.0
            w = cfg["kf_search_window"]
            for idx in range(max(shot.start_frame, center - w),
                             min(shot.end_frame, center + w + 1)):
                frame = _read_frame(cap, idx)
                if frame is None:
                    continue
                blur, brightness = _quality(frame)
                if brightness < cfg["kf_min_brightness"] or brightness > cfg["kf_max_brightness"]:
                    continue  # black / white / fade frame
                if blur > best_blur:
                    best, best_blur = (idx, frame, blur, brightness), blur
            if best is None:
                continue
            idx, frame, blur, brightness = best
            t = idx / fps
            if blur < cfg["kf_min_blur"]:
                continue  # severe motion blur / transition
            if any(abs(t - pt) < cfg["kf_min_gap_sec"] for pt in picked_times):
                continue
            picked_times.append(t)
            kf_id = f"s{shot.index:02d}_f{idx:05d}"
            path = out_dir / f"{kf_id}.png"
            cv2.imwrite(str(path), frame)
            keyframes.append(Keyframe(
                kf_id=kf_id, shot_index=shot.index, frame_idx=idx, time_sec=round(t, 3),
                path=str(path), width=frame.shape[1], height=frame.shape[0],
                blur=round(blur, 2), brightness=round(brightness, 2)))
        if not any(k.shot_index == shot.index for k in keyframes):
            # guarantee at least one keyframe per shot: take best middle frame regardless
            center = shot.start_frame + n // 2
            frame = _read_frame(cap, center)
            if frame is not None:
                blur, brightness = _quality(frame)
                kf_id = f"s{shot.index:02d}_f{center:05d}"
                path = out_dir / f"{kf_id}.png"
                cv2.imwrite(str(path), frame)
                keyframes.append(Keyframe(
                    kf_id=kf_id, shot_index=shot.index, frame_idx=center,
                    time_sec=round(center / fps, 3), path=str(path),
                    width=frame.shape[1], height=frame.shape[0],
                    blur=round(blur, 2), brightness=round(brightness, 2)))
    cap.release()
    return keyframes
