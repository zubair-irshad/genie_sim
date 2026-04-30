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
    return renderer.add_orbit_cameras(
        "demo_cam",
        center=(0.0, 0.0, 0.8),
        radius=2.0,
        height=0.6,
        num_cameras=count,
        resolution=(640, 480),
    )


def pair_id(index: int) -> str:
    return f"{index:04d}"


def relative_to_cwd(path: str | Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except Exception:
        return str(path)
