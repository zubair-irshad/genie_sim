from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.data.image_io import (
    save_png,
    visualize_depth,
    visualize_normals,
    visualize_segmentation,
)
from diffusion_harmonizer.rendering import launch_renderer


def _prefer(paths: list[Path], needles: list[str], count: int = 1) -> list[Path]:
    ranked = []
    for path in paths:
        score = sum(needle.lower() in str(path).lower() for needle in needles)
        ranked.append((score, str(path), path))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:count]]


def add_procedural_table(renderer) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = renderer.stage
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.35))
    table.AddScaleOp().Set(Gf.Vec3f(0.9, 0.6, 0.05))
    mat = UsdShade.Material.Define(stage, "/World/TableMat")
    shader = UsdShade.Shader.Define(stage, "/World/TableMat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.55, 0.52))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(table).Bind(mat)


def build_demo_scene(renderer, index: AssetIndex) -> dict[str, list[str]]:
    backgrounds = index.backgrounds()
    robots = _prefer(index.robots(), ["G2", "genie", "franka"], count=1)
    objects = _prefer(index.objects(), ["cup", "box", "bottle", "can", "fruit", "block"], count=3)
    hdris = index.hdris()

    if backgrounds:
        renderer.open_scene(str(backgrounds[0]))
    add_procedural_table(renderer)

    referenced = {"robots": [], "objects": [], "hdri": []}
    if robots:
        renderer.reference_asset(str(robots[0]), "/World/Robot", translate=(-0.6, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        referenced["robots"].append(str(robots[0]))
    for idx, obj in enumerate(objects):
        x = -0.25 + 0.25 * idx
        renderer.reference_asset(str(obj), f"/World/Object_{idx}", translate=(x, 0.0, 0.47), scale=(1.0, 1.0, 1.0))
        referenced["objects"].append(str(obj))
    if hdris:
        renderer.set_dome_light(str(hdris[0]), intensity=1200.0, rotation_deg=0.0)
        referenced["hdri"].append(str(hdris[0]))
    renderer.set_distant_light(intensity=2500.0, angle_deg=1.5, direction=(-0.4, -0.3, -1.0))
    return referenced


def run(args: argparse.Namespace) -> None:
    index = AssetIndex(args.assets_root)
    if not index.records:
        raise FileNotFoundError(f"No assets indexed under {args.assets_root}. Finish the GenieSimAssets download first.")
    index.write_manifest(Path(args.assets_root) / "manifest.json")

    renderer = launch_renderer(headless=args.headless, renderer_type=args.renderer_type)
    try:
        build_demo_scene(renderer, index)
        cameras = renderer.add_orbit_cameras(
            "phase0_cam",
            center=(0.0, 0.0, 0.55),
            radius=1.8,
            height=0.55,
            num_cameras=5,
            resolution=(args.width, args.height),
        )
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, cam in enumerate(cameras):
            frame = renderer.capture_frame(cam, rgb=True, depth=True, segmentation=True, normal=True)
            rgb = frame["rgb"]
            depth = visualize_depth(frame["depth"])
            seg = visualize_segmentation(frame["segmentation"])
            normal = visualize_normals(frame["normal"])
            save_png(output / f"cam_{idx:03d}_rgb.png", rgb)
            save_png(output / f"cam_{idx:03d}_depth.png", depth)
            save_png(output / f"cam_{idx:03d}_seg.png", seg)
            save_png(output / f"cam_{idx:03d}_normal.png", normal)
            rows.append([rgb, depth, seg, normal])

        fig, axes = plt.subplots(5, 4, figsize=(12, 12))
        titles = ["RGB", "Depth", "Instance Seg", "Normals"]
        for r in range(5):
            for c in range(4):
                axes[r, c].imshow(rows[r][c])
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(titles[c])
        fig.tight_layout()
        fig.savefig(output / "phase0_grid.png", dpi=160)
        plt.close(fig)
    finally:
        renderer.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_root", default="assets/geniesim")
    parser.add_argument("--output_dir", default="data/demo_phase0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--renderer_type", default="raytraced")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
