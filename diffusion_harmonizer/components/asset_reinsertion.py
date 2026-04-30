from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import feather_mask, foreground_mask, pair_id
from diffusion_harmonizer.data.image_io import save_png, write_pair


def train_background_gs(dataset_dir: str | Path, output_path: str | Path, gsplat_command: str | None) -> Path:
    if not gsplat_command:
        raise RuntimeError("No gsplat training command configured. Pass --gsplat_command for asset re-insertion.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([gsplat_command, "--data", str(dataset_dir), "--output", str(output), "--strategy", "background_full"], check=True)
    return output


def render_background_gs(checkpoint: str | Path, camera_json: str | Path, output_dir: str | Path, gsplat_render_command: str | None) -> Path:
    if not gsplat_render_command:
        raise RuntimeError("No gsplat render command configured. Pass --gsplat_render_command for asset re-insertion.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [gsplat_render_command, "--checkpoint", str(checkpoint), "--cameras", str(camera_json), "--output", str(output)],
        check=True,
    )
    return output


def generate_pairs(
    renderer,
    output_dir: str | Path = "data/asset_reinsertion/demo",
    count: int = 30,
    inserted_paths: list[str] | None = None,
    gsplat_command: str | None = None,
    gsplat_render_command: str | None = None,
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    del seed
    from diffusion_harmonizer.components.artifacts_correction import export_capture_dataset

    output = Path(output_dir)
    inserted_paths = inserted_paths or ["inserted", "object", "Obj"]
    sphere = renderer.add_sphere_cameras("reinsertion_bg", center=(0.0, 0.0, 0.8), radius=2.0, num_cameras=100)
    renderer.set_path_tracing(True, spp=64)
    bg_frames = renderer.capture_multiview(sphere, rgb=True, depth=True)
    dataset_dir = export_capture_dataset(bg_frames, output / "empty_background_capture")
    bg_checkpoint = train_background_gs(dataset_dir, output / "scene_bg_gs", gsplat_command)
    bg_render_dir = render_background_gs(bg_checkpoint, dataset_dir / "cameras.json", output / "bg_renders", gsplat_render_command)

    entries: dict[str, dict[str, str]] = {}
    import imageio.v3 as iio

    for idx, camera in enumerate(sphere[:count]):
        renderer.set_shadows_enabled(True)
        renderer.set_path_tracing(True, spp=64)
        target_frame = renderer.capture_frame(camera, rgb=True, segmentation=True)
        target = target_frame["rgb"]
        mask = foreground_mask(target_frame["segmentation"], target_frame["segmentation_mapping"] or {}, inserted_paths)
        mask = feather_mask(mask, sigma=2.0)

        renderer.set_shadows_enabled(False)
        # In production, hide non-inserted prims before this capture to obtain a
        # transparent foreground pass. The alpha still comes from Replicator.
        fg_frame = renderer.capture_frame(camera, rgb=True)
        renderer.set_shadows_enabled(True)
        fg = fg_frame["rgb"]
        bg_path = bg_render_dir / f"{idx:04d}.png"
        if not bg_path.exists():
            continue
        bg = np.asarray(iio.imread(bg_path))[..., :3]
        degraded = (mask[..., None] * fg.astype(np.float32) + (1.0 - mask[..., None]) * bg.astype(np.float32)).astype(np.uint8)
        pair_dir = output / pair_id(idx)
        save_png(pair_dir / "bg_render.png", bg)
        save_png(pair_dir / "fg_only.png", fg)
        key = f"asset_reinsertion_{pair_id(idx)}"
        entries[key] = write_pair(
            pair_dir,
            degraded,
            target,
            {
                "component": "asset_reinsertion",
                "camera": camera,
                "gs_checkpoint": str(bg_checkpoint),
                "inserted_path_needles": inserted_paths,
                "note": "Static simulator background capture replaces 3DGUT; foreground is USD asset state.",
            },
            mask=mask,
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/asset_reinsertion/demo")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--gsplat_command", default=None)
    parser.add_argument("--gsplat_render_command", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        generate_pairs(
            renderer,
            args.output_dir,
            args.count,
            gsplat_command=args.gsplat_command,
            gsplat_render_command=args.gsplat_render_command,
            seed=args.seed,
        )
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
