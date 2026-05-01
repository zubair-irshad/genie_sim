from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.components.common import discover_demo_cameras, pair_id
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


def generate_pairs(
    renderer,
    assets_root: str | Path = "assets/geniesim",
    output_dir: str | Path = "data/shadow_simulation/demo",
    count: int = 30,
    seed: int = 42,
    pre_pair_callback=None,
    hdri_query: str | None = "indoor,studio,kitchen,office,warehouse,room",
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    index = AssetIndex(assets_root)
    hdris = _filtered_hdris(index, hdri_query)
    if not hdris:
        raise FileNotFoundError(f"No HDRI files found under {assets_root}; finish the GenieSimAssets download first.")

    output = Path(output_dir)
    cameras = discover_demo_cameras(renderer, count=min(5, count))
    entries: dict[str, dict[str, str]] = {}
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
        target = renderer.capture_frame(camera, rgb=True)["rgb"]

        # Keep lighting fixed. Component 4 should isolate missing/weak shadow
        # artifacts; foreground/background relighting belongs to Component 3.
        renderer.set_dome_light(hdri, intensity=dome_intensity, rotation_deg=dome_rotation)
        renderer.set_distant_light(intensity=sun["intensity"], angle_deg=sun["angle_deg"], direction=sun["direction"])
        renderer.set_shadows_enabled(False)
        renderer.set_path_tracing(True, spp=64)
        degraded = renderer.capture_frame(camera, rgb=True)["rgb"]
        renderer.set_shadows_enabled(True)

        diff = np.abs(target.astype(np.int16) - degraded.astype(np.int16)).astype(np.uint8)
        pair_dir = output / pair_id(idx)
        save_png(pair_dir / "shadow_diff.png", diff)
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
                "degradation": "same lights with RTX shadows disabled",
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
