from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_phase0 import build_demo_scene
from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.components import isp_modification, shadow_simulation
from diffusion_harmonizer.components.scene_randomization import Phase1SceneRandomizer
from diffusion_harmonizer.rendering import launch_renderer


def write_report(output: Path, master: dict[str, dict[str, dict[str, str]]]) -> None:
    rows = []
    for key, item in master["train"].items():
        comparison = Path(item["image"]).parent / "comparison.png"
        rows.append(
            "<figure>"
            f"<img src='{comparison.as_posix()}' loading='lazy' />"
            f"<figcaption>{key}</figcaption>"
            "</figure>"
        )
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Genie Sim DiffusionHarmonizer Demo Pairs</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #f6f6f4; color: #1d1d1b; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
    figure { margin: 0; background: white; border: 1px solid #ddd; padding: 8px; }
    img { width: 100%; display: block; }
    figcaption { font-size: 13px; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>Genie Sim DiffusionHarmonizer Demo Pairs</h1>
  <div class="grid">
""" + "\n".join(rows) + """
  </div>
</body>
</html>
"""
    (output / "report.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Genie Sim demo pairs for DiffusionHarmonizer components.")
    parser.add_argument("--assets_root", default="assets/geniesim")
    parser.add_argument("--output_dir", default="data/demo")
    parser.add_argument("--robot_query", default="franka")
    parser.add_argument("--robot_usd", default=None, help="Explicit Franka/Panda USD path. Use this when assets_root is GenieSimAssets but Franka lives elsewhere.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--isp_count", type=int, default=30)
    parser.add_argument("--shadow_count", type=int, default=30)
    parser.add_argument("--gsplat_command", default=None)
    parser.add_argument("--gsplat_render_command", default=None)
    parser.add_argument("--relighting_command", default=None)
    parser.add_argument("--include_external", action="store_true", help="Also run relighting and 3DGS components that require sidecars.")
    parser.add_argument("--use_background", action="store_true", help="Open a full indoor background USD if one can be identified.")
    args = parser.parse_args()

    index = AssetIndex(args.assets_root)
    if not index.records:
        raise FileNotFoundError(f"No assets indexed under {args.assets_root}. Finish the GenieSimAssets download first.")
    index.write_manifest(Path(args.assets_root) / "manifest.json")

    renderer = launch_renderer(headless=True)
    master: dict[str, dict[str, dict[str, str]]] = {"train": {}}
    output = Path(args.output_dir)
    try:
        referenced = build_demo_scene(
            renderer,
            index,
            robot_query=args.robot_query,
            robot_usd=args.robot_usd,
            use_background=args.use_background,
        )
        print(f"Phase 1 referenced assets: {referenced}")
        randomize_scene = Phase1SceneRandomizer(
            renderer,
            robot_prim_path="/World/Robot",
            object_prim_paths=referenced.get("object_prim_paths", []),
            seed=args.seed,
        )
        master["train"].update(
            isp_modification.generate_pairs(
                renderer,
                output / "isp_modification" / "demo",
                count=args.isp_count,
                foreground_paths=["Robot", "Object_", "franka", "panda", "object"],
                seed=args.seed,
                pre_pair_callback=randomize_scene,
            )
        )
        master["train"].update(
            shadow_simulation.generate_pairs(
                renderer,
                assets_root=args.assets_root,
                output_dir=output / "shadow_simulation" / "demo",
                count=args.shadow_count,
                seed=args.seed,
                pre_pair_callback=randomize_scene,
            )
        )
        if args.include_external:
            from diffusion_harmonizer.components import artifacts_correction, asset_reinsertion, relighting

            master["train"].update(
                relighting.generate_pairs(
                    renderer,
                    output / "relighting" / "demo",
                    count=30,
                    relighting_command=args.relighting_command,
                    seed=args.seed,
                )
            )
            master["train"].update(
                artifacts_correction.generate_pairs(
                    renderer,
                    output / "artifacts_correction" / "demo",
                    count=50,
                    gsplat_command=args.gsplat_command,
                    seed=args.seed,
                )
            )
            master["train"].update(
                asset_reinsertion.generate_pairs(
                    renderer,
                    output / "asset_reinsertion" / "demo",
                    count=30,
                    gsplat_command=args.gsplat_command,
                    gsplat_render_command=args.gsplat_render_command,
                    seed=args.seed,
                )
            )
        (output / "demo_pairs.json").write_text(json.dumps(master, indent=2))
        write_report(output, master)
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
