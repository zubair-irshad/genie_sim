from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import discover_demo_cameras, feather_mask, foreground_mask, pair_id
from diffusion_harmonizer.data.image_io import write_pair


@dataclass
class ISPParams:
    exposure_ev: float
    white_balance_K: int
    gamma: float
    saturation: float
    contrast: float
    hue_shift: float
    noise_sigma: float


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    x = np.clip(image, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def sample_isp_params(rng: random.Random, scale: float = 1.0) -> ISPParams:
    return ISPParams(
        exposure_ev=rng.uniform(-1.5, 1.5) * scale,
        white_balance_K=int(rng.uniform(3000, 8000)),
        gamma=1.0 + (rng.uniform(0.6, 1.8) - 1.0) * scale,
        saturation=1.0 + (rng.uniform(0.5, 1.5) - 1.0) * scale,
        contrast=1.0 + (rng.uniform(0.7, 1.3) - 1.0) * scale,
        hue_shift=rng.uniform(-15.0, 15.0) * scale,
        noise_sigma=rng.uniform(0.0, 10.0) * scale,
    )


def _kelvin_rgb(kelvin: int) -> np.ndarray:
    t = kelvin / 100.0
    if t <= 66:
        red = 255.0
        green = 99.4708025861 * np.log(t) - 161.1195681661
        blue = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    else:
        red = 329.698727446 * ((t - 60) ** -0.1332047592)
        green = 288.1221695283 * ((t - 60) ** -0.0755148492)
        blue = 255.0
    return np.clip([red, green, blue], 1.0, 255.0).astype(np.float32) / 255.0


def apply_software_isp(image: np.ndarray, params: ISPParams, rng: random.Random | None = None) -> np.ndarray:
    import cv2

    rng = rng or random.Random()
    lin = srgb_to_linear(image)
    lin *= 2.0 ** params.exposure_ev
    wb = _kelvin_rgb(params.white_balance_K)
    lin *= wb / max(float(np.mean(wb)), 1e-6)
    srgb = linear_to_srgb(lin).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(srgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + params.hue_shift / 2.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * params.saturation, 0.0, 1.0)
    srgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    srgb = np.clip((srgb - 0.5) * params.contrast + 0.5, 0.0, 1.0)
    srgb = np.clip(np.power(np.clip(srgb, 0.0, 1.0), 1.0 / max(params.gamma, 1e-3)), 0.0, 1.0)
    if params.noise_sigma > 0:
        noise = np.asarray([rng.gauss(0.0, params.noise_sigma / 255.0) for _ in range(srgb.size)], dtype=np.float32)
        srgb = np.clip(srgb + noise.reshape(srgb.shape), 0.0, 1.0)
    return (srgb * 255.0).astype(np.uint8)


def generate_pairs(
    renderer,
    output_dir: str | Path = "data/isp_modification/demo",
    count: int = 30,
    foreground_paths: list[str] | None = None,
    seed: int = 42,
    pre_pair_callback=None,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    output = Path(output_dir)
    cameras = discover_demo_cameras(renderer, count=min(5, count))
    entries: dict[str, dict[str, str]] = {}
    foreground_paths = foreground_paths or ["robot", "object", "Obj", "G2", "Franka"]

    for idx in range(count):
        camera = cameras[idx % len(cameras)]
        scene_state = pre_pair_callback(idx, camera) if pre_pair_callback else {}
        frame = renderer.capture_frame(camera, rgb=True, segmentation=True)
        target = frame["rgb"]
        params = sample_isp_params(rng, scale=0.3 if idx >= int(count * 0.67) else 1.0)
        isp = apply_software_isp(target, params, rng)
        if idx >= int(count * 0.67):
            mask = np.ones(target.shape[:2], dtype=np.float32)
            mode = "full_frame_mild"
        else:
            mask = foreground_mask(frame["segmentation"], frame["segmentation_mapping"] or {}, foreground_paths)
            mask = feather_mask(mask, sigma=3.0)
            mode = "masked_foreground"
        mixed = (mask[..., None] * isp.astype(np.float32) + (1.0 - mask[..., None]) * target.astype(np.float32)).astype(np.uint8)
        key = f"isp_{pair_id(idx)}"
        entries[key] = write_pair(
            output / pair_id(idx),
            mixed,
            target,
            {
                "component": "isp_modification",
                "mode": mode,
                "camera": camera,
                "params": asdict(params),
                "scene_state": scene_state,
            },
            mask=mask,
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/isp_modification/demo")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        generate_pairs(renderer, args.output_dir, args.count, seed=args.seed)
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
