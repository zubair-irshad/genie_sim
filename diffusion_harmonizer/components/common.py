from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ids_matching_paths(mapping: dict[int, str], path_needles: Iterable[str]) -> list[int]:
    needles = [needle.lower() for needle in path_needles if needle]
    if not needles:
        return [idx for idx in mapping if idx != 0]
    matched = []
    for idx, path in mapping.items():
        low = path.lower()
        if any(needle in low for needle in needles):
            matched.append(idx)
    return matched


def foreground_mask(segmentation: np.ndarray, mapping: dict[int, str], path_needles: Iterable[str]) -> np.ndarray:
    ids = ids_matching_paths(mapping, path_needles)
    if not ids:
        return np.zeros(segmentation.shape[:2], dtype=np.float32)
    return np.isin(segmentation, ids).astype(np.float32)


def feather_mask(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)


def discover_demo_cameras(renderer, count: int = 5) -> list[str]:
    if renderer.cameras:
        return list(renderer.cameras.keys())[:count]
    presets = [
        ((1.7, -1.2, 1.35), (-0.20, 0.0, 0.82)),
        ((1.6, 1.2, 1.25), (-0.20, 0.0, 0.82)),
        ((-1.7, -1.1, 1.35), (-0.35, 0.0, 0.95)),
        ((-1.8, 1.0, 1.25), (-0.35, 0.0, 0.95)),
        ((0.0, -2.0, 1.55), (-0.15, 0.0, 0.88)),
        ((-2.2, 0.0, 1.65), (-0.55, 0.0, 1.0)),
    ]
    names = []
    for idx in range(count):
        position, look_at = presets[idx % len(presets)]
        name = f"demo_cam_{idx:03d}"
        renderer.add_camera(name, position=position, look_at=look_at, resolution=(640, 480), focal_length=20.0)
        names.append(name)
    return names


def pair_id(index: int) -> str:
    return f"{index:04d}"


def relative_to_cwd(path: str | Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except Exception:
        return str(path)
