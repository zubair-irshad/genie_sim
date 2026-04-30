from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_harmonizer.components.common import pair_id
from diffusion_harmonizer.data.image_io import save_png, write_pair


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray, world_t_cam: np.ndarray, stride: int = 16) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (xs[valid] - cx) * z[valid] / fx
    y = (ys[valid] - cy) * z[valid] / fy
    cam = np.stack([x, y, -z[valid], np.ones_like(z[valid])], axis=-1)
    world = (world_t_cam @ cam.T).T[:, :3]
    return world.astype(np.float32)


def export_capture_dataset(frames: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Export privileged sphere captures to a simple gsplat-friendly dataset.

    The exported JSON contains exact intrinsics/extrinsics and dense depth-derived
    points. If a COLMAP converter is available later, this metadata is enough to
    emit `sparse/0/*.bin` without doing SfM. The privileged 100-camera static
    capture replaces 3DGUT because the simulator controls physics and viewpoints.
    """

    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    points = []
    cameras = []
    for idx, frame in enumerate(frames):
        image_name = f"{idx:04d}.png"
        save_png(images_dir / image_name, frame["rgb"])
        intrinsics = np.asarray(frame["camera_intrinsics"], dtype=float)
        extrinsics = np.asarray(frame["camera_extrinsics"], dtype=float)
        if frame.get("depth") is not None:
            points.append(backproject_depth(frame["depth"], intrinsics, extrinsics))
        cameras.append(
            {
                "image": image_name,
                "width": int(frame["rgb"].shape[1]),
                "height": int(frame["rgb"].shape[0]),
                "intrinsics": intrinsics.tolist(),
                "world_T_cam": extrinsics.tolist(),
            }
        )
    point_cloud = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float32)
    np.save(output / "points3d.npy", point_cloud)
    (output / "cameras.json").write_text(json.dumps({"cameras": cameras, "points3d_npy": "points3d.npy"}, indent=2))
    return output


def run_gsplat_strategy(
    dataset_dir: str | Path,
    output_dir: str | Path,
    strategy: str,
    gsplat_command: str | None = None,
    iterations: int = 30000,
) -> Path:
    """Run an external gsplat/nerfstudio strategy command.

    The repository vendors gsplat examples, but CUDA-compatible training is
    environment-specific. This function provides the stable integration point
    used by the demo runner.
    """

    if not gsplat_command:
        raise RuntimeError("No gsplat training command configured. Pass --gsplat_command for artifact generation.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            gsplat_command,
            "--data",
            str(dataset_dir),
            "--output",
            str(output),
            "--strategy",
            strategy,
            "--iterations",
            str(iterations),
        ],
        check=True,
    )
    return output


def generate_pairs(
    renderer,
    output_dir: str | Path = "data/artifacts_correction/demo",
    count: int = 50,
    gsplat_command: str | None = None,
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    del seed
    output = Path(output_dir)
    cameras = renderer.add_sphere_cameras("artifacts_sphere", center=(0.0, 0.0, 0.8), radius=2.0, num_cameras=100)
    renderer.set_path_tracing(True, spp=64)
    frames = renderer.capture_multiview(cameras, rgb=True, depth=True)
    dataset_dir = export_capture_dataset(frames, output / "privileged_capture")
    # Strategies A and D are required for the demo; B/C can be added by passing
    # strategy names understood by the sidecar trainer.
    run_gsplat_strategy(dataset_dir, output / "gs_sparse_k2", "sparse_k2", gsplat_command=gsplat_command, iterations=30000)
    run_gsplat_strategy(dataset_dir, output / "gs_underfit_25", "underfit_25", gsplat_command=gsplat_command, iterations=7500)

    # The sidecar renderer is expected to write degraded renders named by view.
    entries: dict[str, dict[str, str]] = {}
    degraded_roots = [output / "gs_sparse_k2" / "renders", output / "gs_underfit_25" / "renders"]
    idx = 0
    import imageio.v3 as iio

    for root in degraded_roots:
        for degraded_path in sorted(root.glob("*.png")):
            if idx >= count:
                return entries
            view_idx = int(degraded_path.stem)
            degraded = np.asarray(iio.imread(degraded_path))[..., :3]
            target = frames[view_idx]["rgb"]
            key = f"artifacts_{pair_id(idx)}"
            entries[key] = write_pair(
                output / pair_id(idx),
                degraded,
                target,
                {"component": "artifacts_correction", "strategy": root.parent.name, "view_index": view_idx},
            )
            idx += 1
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/artifacts_correction/demo")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--gsplat_command", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        generate_pairs(renderer, args.output_dir, args.count, args.gsplat_command, args.seed)
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
