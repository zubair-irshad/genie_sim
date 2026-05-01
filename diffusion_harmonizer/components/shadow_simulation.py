from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.components.common import discover_demo_cameras, feather_mask, foreground_mask, pair_id
from diffusion_harmonizer.data.image_io import save_png, write_pair


def _random_unit_vector_above_horizon(rng: random.Random) -> tuple[float, float, float]:
    az = rng.uniform(0.0, 2.0 * np.pi)
    elev = rng.uniform(np.radians(15.0), np.radians(75.0))
    return (
        float(np.cos(elev) * np.cos(az)),
        float(np.cos(elev) * np.sin(az)),
        float(-np.sin(elev)),
    )


def _filtered_hdris(index: AssetIndex, query: str | None = None) -> list[Path]:
    hdris = index.hdris()
    if not query:
        return hdris
    include = [token.strip().lower() for token in query.split(",") if token.strip()]
    bad = ("outdoor", "forest", "field", "park", "street", "road", "airport", "beach", "mountain", "sky")
    ranked = []
    for path in hdris:
        low = str(path).lower()
        if any(token in low for token in bad) and not any(token in low for token in include):
            continue
        score = sum(token in low for token in include)
        ranked.append((score, str(path), path))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked] or hdris


def _attenuate_cast_shadows(image: np.ndarray, foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fallback when RTX shadow toggles do not remove shadows reliably.

    This estimates dark cast-shadow regions on low-saturation tabletop/background
    receivers and lifts only those pixels. Foreground assets are excluded.
    """

    import cv2

    rgb = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    local_light = cv2.GaussianBlur(value, (0, 0), sigmaX=21.0, sigmaY=21.0)
    shadow_score = np.clip((local_light - value - 0.045) / 0.22, 0.0, 1.0)
    receiver = ((saturation < 0.42) & (local_light > 0.28)).astype(np.float32)
    shadow_mask = shadow_score * receiver * (1.0 - np.clip(foreground, 0.0, 1.0))
    shadow_mask = feather_mask(shadow_mask, sigma=4.0)
    lifted = rgb + shadow_mask[..., None] * (local_light[..., None] - rgb) * 0.9
    return (np.clip(lifted, 0.0, 1.0) * 255.0).astype(np.uint8), np.clip(shadow_mask, 0.0, 1.0)


def generate_pairs(
    renderer,
    assets_root: str | Path = "assets/geniesim",
    output_dir: str | Path = "data/shadow_simulation/demo",
    count: int = 30,
    seed: int = 42,
    pre_pair_callback=None,
    hdri_query: str | None = "indoor,studio,kitchen,office,warehouse,room",
    foreground_paths: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    index = AssetIndex(assets_root)
    hdris = _filtered_hdris(index, hdri_query)
    if not hdris:
        raise FileNotFoundError(f"No HDRI files found under {assets_root}; finish the GenieSimAssets download first.")

    output = Path(output_dir)
    cameras = discover_demo_cameras(renderer, count=min(5, count))
    entries: dict[str, dict[str, str]] = {}
    foreground_paths = foreground_paths or ["/World/Robot", "/World/Object_"]
    for idx in range(count):
        camera = cameras[idx % len(cameras)]
        scene_state = pre_pair_callback(idx, camera) if pre_pair_callback else {}
        hdri = str(rng.choice(hdris))
        dome_intensity = rng.uniform(800.0, 1600.0)
        dome_rotation = rng.uniform(0.0, 360.0)
        sun = {
            "intensity": rng.uniform(500.0, 1500.0),
            "angle_deg": rng.uniform(2.0, 8.0),
            "direction": _random_unit_vector_above_horizon(rng),
        }

        renderer.set_dome_light(hdri, intensity=dome_intensity, rotation_deg=dome_rotation)
        renderer.set_distant_light(intensity=sun["intensity"], angle_deg=sun["angle_deg"], direction=sun["direction"])
        renderer.set_shadows_enabled(True)
        renderer.set_path_tracing(True, spp=64)
        target_frame = renderer.capture_frame(camera, rgb=True, segmentation=True)
        target = target_frame["rgb"]
        fg_mask = foreground_mask(
            target_frame["segmentation"],
            target_frame["segmentation_mapping"] or {},
            foreground_paths,
        )
        fg_mask = feather_mask(fg_mask, sigma=2.0)

        # Keep lighting fixed. Component 4 should isolate missing/weak shadow
        # artifacts; foreground/background relighting belongs to Component 3.
        renderer.set_dome_light(hdri, intensity=dome_intensity, rotation_deg=dome_rotation)
        renderer.set_distant_light(intensity=sun["intensity"], angle_deg=sun["angle_deg"], direction=sun["direction"])
        renderer.set_shadows_enabled(False)
        renderer.set_path_tracing(True, spp=64)
        degraded = renderer.capture_frame(camera, rgb=True)["rgb"]
        renderer.set_shadows_enabled(True)

        diff = np.abs(target.astype(np.int16) - degraded.astype(np.int16)).astype(np.uint8)
        attenuated, shadow_mask = _attenuate_cast_shadows(target, fg_mask)
        shadow_region = shadow_mask > 0.05
        non_shadow_region = shadow_mask <= 0.02
        render_shadow_delta = float(np.mean(diff[shadow_region])) if np.any(shadow_region) else 0.0
        render_non_shadow_delta = float(np.mean(diff[non_shadow_region])) if np.any(non_shadow_region) else float(np.mean(diff))
        fallback_used = render_shadow_delta < 8.0 or render_non_shadow_delta > 6.0
        if fallback_used:
            degraded = attenuated
            diff = np.abs(target.astype(np.int16) - degraded.astype(np.int16)).astype(np.uint8)
        pair_dir = output / pair_id(idx)
        save_png(pair_dir / "shadow_diff.png", diff)
        save_png(pair_dir / "shadow_mask.png", np.repeat((shadow_mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
        key = f"shadow_{pair_id(idx)}"
        entries[key] = write_pair(
            pair_dir,
            degraded,
            target,
            {
                "component": "shadow_simulation",
                "camera": camera,
                "hdri": hdri,
                "dome_intensity": dome_intensity,
                "dome_rotation_deg": dome_rotation,
                "distant_light": sun,
                "degradation": "same lights with cast shadows disabled or attenuated",
                "shadow_attenuation_fallback": fallback_used,
                "render_shadow_delta": render_shadow_delta,
                "render_non_shadow_delta": render_non_shadow_delta,
                "scene_state": scene_state,
                "shadow_toggle_settings": [
                    "/rtx/shadows/enabled",
                    "/rtx/directLighting/shadows/enabled",
                    "/rtx/raytracing/shadows/enabled",
                    "/persistent/rtx/shadows/enabled",
                ],
            },
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_root", default="assets/geniesim")
    parser.add_argument("--output_dir", default="data/shadow_simulation/demo")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdri_query", default="indoor,studio,kitchen,office,warehouse,room")
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        generate_pairs(renderer, args.assets_root, args.output_dir, args.count, args.seed, hdri_query=args.hdri_query)
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
