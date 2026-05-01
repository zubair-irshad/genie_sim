# Robotics DiffusionHarmonizer on Genie Sim

This repo path assumes Genie Sim 3.x on Isaac Sim 5.1. Run all scripts with Isaac Sim's bundled Python, not the system interpreter.

## Runtime

Host install:

```bash
export ISAACSIM_PATH=/path/to/isaac-sim
export PYTHON=$ISAACSIM_PATH/python.sh
$PYTHON -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()"
```

Container install:

```bash
export PYTHON=/isaac-sim/python.sh
$PYTHON -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()"
```

Genie Sim 3.0/3.1 is expected to run against Isaac Sim 5.1 and Python 3.11. Standalone scripts must instantiate `SimulationApp` before any `omni.*`, `pxr.*`, or `isaacsim.*` imports. `diffusion_harmonizer.rendering.launch_renderer()` enforces that ordering.

## Python Extras

Install extras into Isaac Sim's Python:

```bash
$PYTHON -m pip install huggingface_hub modelscope tqdm requests opencv-python-headless imageio matplotlib gsplat
```

`gsplat` may need to match the CUDA/PyTorch build shipped with Isaac Sim. This repo also vendors gsplat sources at `source/scene_reconstruction/third_party/gsplat`.

## GenieSimAssets Download

The assets are a download, not a curation task. Put them under `assets/geniesim/`.

```bash
$PYTHON - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "agibot-world/GenieSimAssets",
    repo_type="dataset",
    local_dir="assets/geniesim/",
    # allow_patterns=["objects/**", "materials/hdri/**", "robot/**", "background/**"],
)
PY
```

Fallback if HuggingFace is slow:

```bash
$PYTHON - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("agibot_world/GenieSimAssets", local_dir="assets/geniesim/")
PY
```

Build the manifest:

```bash
$PYTHON -m diffusion_harmonizer.asset_manager.asset_index
```

or let `demo_phase0.py` write `assets/geniesim/manifest.json`.

## Phase 0 Demo

```bash
$PYTHON demo_phase0.py --assets_root assets/geniesim --output_dir data/demo_phase0
```

Expected output:

```text
data/demo_phase0/
  cam_000_rgb.png
  cam_000_depth.png
  cam_000_seg.png
  cam_000_normal.png
  ...
  phase0_grid.png
```

## Phase 1 Demo

By default the demo runner now generates only the two self-contained components
that do not need sidecar model/training processes: ISP modification and shadow
simulation.

```bash
$PYTHON demo_paired_data.py --assets_root assets/geniesim --output_dir data/demo
```

If the Franka/Panda USD lives outside the asset root, pass it explicitly:

```bash
$PYTHON demo_paired_data.py \
  --assets_root GenieSimAssets \
  --robot_usd assets/roboverse_data/robots/franka/usd/franka_v2.usd \
  --output_dir data/demo
```

For each ISP/shadow pair the runner applies one of several valid Franka arm
configurations and jitters the foreground objects on the tabletop. The applied
configuration is recorded in each pair's `metadata.json`.

Components that need external models accept sidecar commands:

```bash
$PYTHON demo_paired_data.py \
  --include_external \
  --gsplat_command /path/to/train_gs_strategy \
  --gsplat_render_command /path/to/render_gs \
  --relighting_command /path/to/relight_diffusion
```

The privileged 100-camera sphere replaces 3DGUT for Components 1 and 5. We control the simulator, so physics can be paused, the empty static background can be captured directly, and foregrounds are known USD assets with exact masks from Replicator instance segmentation.
