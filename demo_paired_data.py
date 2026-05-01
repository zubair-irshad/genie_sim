from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_phase0 import build_demo_scene
from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.components import isp_modification, shadow_simulation
from diffusion_harmonizer.components.scene_randomization import Phase1SceneRandomizer
from diffusion_harmonizer.rendering import launch_renderer


def log(message: str) -> None:
    print(f"[demo_paired_data] {message}", flush=True)


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
    parser.add_argument("--hdri_query", default="indoor,studio,kitchen,office,warehouse,room")
    parser.add_argument("--table_height", type=float, default=0.72)
    parser.add_argument("--object_z", type=float, default=None)
    parser.add_argument("--object_scale", type=float, default=0.45)
    parser.add_argument("--robot_x", type=float, default=-0.95)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--robot_z", type=float, default=0.72)
    parser.add_argument("--robot_yaw", type=float, default=0.0)
    parser.add_argument("--robot_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--isp_count", type=int, default=30)
    parser.add_argument("--shadow_count", type=int, default=30)
    parser.add_argument("--gsplat_command", default=None)
    parser.add_argument("--gsplat_render_command", default=None)
    parser.add_argument("--relighting_command", default=None)
    parser.add_argument("--include_external", action="store_true", help="Also run relighting and 3DGS components that require sidecars.")
    parser.add_argument("--use_background", action="store_true", help="Open a full indoor background USD if one can be identified.")
    args = parser.parse_args()

    if args.robot_usd and not Path(args.robot_usd).exists():
        raise FileNotFoundError(f"--robot_usd does not exist: {args.robot_usd}")

    log(f"Indexing assets under {args.assets_root}")
    index = AssetIndex(args.assets_root)
    if not index.records:
        raise FileNotFoundError(f"No assets indexed under {args.assets_root}. Finish the GenieSimAssets download first.")
    index.write_manifest(Path(args.assets_root) / "manifest.json")
    log(f"Indexed {len(index.records)} assets")

    log("Launching Isaac Sim renderer")
    renderer = launch_renderer(headless=True)
    master: dict[str, dict[str, dict[str, str]]] = {"train": {}}
    output = Path(args.output_dir)
    try:
        log("Building tabletop scene")
        referenced = build_demo_scene(
            renderer,
            index,
            robot_query=args.robot_query,
            robot_usd=args.robot_usd,
            use_background=args.use_background,
            hdri_query=args.hdri_query,
            table_height=args.table_height,
            object_z=args.object_z,
            object_scale=args.object_scale,
            robot_translate=(args.robot_x, args.robot_y, args.robot_z),
            robot_rotate=(0.0, 0.0, args.robot_yaw),
            robot_scale=args.robot_scale,
            log_fn=log,
        )
        log(f"Referenced assets: {referenced}")
        randomize_scene = Phase1SceneRandomizer(
            renderer,
            robot_prim_path="/World/Robot",
            object_prim_paths=referenced.get("object_prim_paths", []),
            object_z=float(referenced["object_z"][0]),
            object_scale=args.object_scale,
            seed=args.seed,
        )
        if args.isp_count > 0:
            log(f"Generating {args.isp_count} ISP pairs")
            isp_entries = isp_modification.generate_pairs(
                renderer,
                output / "isp_modification" / "demo",
                count=args.isp_count,
                foreground_paths=["Robot", "Object_", "franka", "panda", "object"],
                seed=args.seed,
                pre_pair_callback=randomize_scene,
            )
            master["train"].update(isp_entries)
            log(f"Finished ISP pairs: {len(isp_entries)}")
        if args.shadow_count > 0:
            log(f"Generating {args.shadow_count} shadow pairs")
            shadow_entries = shadow_simulation.generate_pairs(
                renderer,
                assets_root=args.assets_root,
                output_dir=output / "shadow_simulation" / "demo",
                count=args.shadow_count,
                seed=args.seed,
                pre_pair_callback=randomize_scene,
            )
            master["train"].update(shadow_entries)
            log(f"Finished shadow pairs: {len(shadow_entries)}")
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
        if not master["train"]:
            raise RuntimeError("No pairs were generated. Check --isp_count, --shadow_count, and asset paths.")
        output.mkdir(parents=True, exist_ok=True)
        (output / "demo_pairs.json").write_text(json.dumps(master, indent=2))
        write_report(output, master)
        log(f"Wrote {output / 'demo_pairs.json'} and {output / 'report.html'}")
    finally:
        log("Shutting down renderer")
        renderer.shutdown()


if __name__ == "__main__":
    main()
