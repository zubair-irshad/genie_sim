from __future__ import annotations

import random
from dataclasses import dataclass


FRANKA_CONFIGS = [
    [0.0, -0.55, 0.0, -2.15, 0.0, 1.65, 0.78],
    [0.35, -0.75, 0.25, -2.35, -0.15, 1.85, 1.05],
    [-0.45, -0.65, -0.2, -2.05, 0.25, 1.55, 0.45],
    [0.65, -0.45, -0.35, -1.85, 0.35, 1.35, 1.25],
    [-0.25, -0.95, 0.55, -2.45, -0.35, 2.0, 0.15],
    [0.15, -0.35, -0.55, -1.75, 0.45, 1.25, 0.95],
]


FRANKA_JOINT_NAME_SETS = [
    [f"panda_joint{i}" for i in range(1, 8)],
    [f"franka_joint{i}" for i in range(1, 8)],
    [f"joint{i}" for i in range(1, 8)],
]


@dataclass
class SceneState:
    robot_config_index: int
    robot_joints: dict[str, float]
    applied_joints: dict[str, float]
    object_placements: list[dict[str, object]]


class Phase1SceneRandomizer:
    """Small indoor tabletop randomizer for ISP and shadow demo pairs."""

    def __init__(
        self,
        renderer,
        robot_prim_path: str = "/World/Robot",
        object_prim_paths: list[str] | None = None,
        object_z: float = 0.722,
        object_scale: float = 0.8,
        seed: int = 42,
    ):
        self.renderer = renderer
        self.robot_prim_path = robot_prim_path
        self.object_prim_paths = object_prim_paths or []
        self.object_z = object_z
        self.object_scale = object_scale
        self.rng = random.Random(seed)

    def __call__(self, pair_index: int, camera_name: str) -> dict[str, object]:
        del camera_name
        config_index = pair_index % len(FRANKA_CONFIGS)
        values = FRANKA_CONFIGS[config_index]
        candidates = [
            dict(zip(names, values))
            for names in FRANKA_JOINT_NAME_SETS
        ]
        applied = {}
        authored = candidates[0]
        for candidate in candidates:
            applied = self.renderer.set_articulation_joint_positions(self.robot_prim_path, candidate)
            if applied:
                authored = candidate
                break

        placements = []
        for idx, prim_path in enumerate(self.object_prim_paths):
            x = -0.25 + 0.25 * idx + self.rng.uniform(-0.06, 0.06)
            y = self.rng.uniform(-0.16, 0.16)
            yaw = self.rng.uniform(-35.0, 35.0)
            z = self.object_z
            self.renderer.set_prim_transform(
                prim_path,
                translate=(x, y, z),
                rotate=(0.0, 0.0, yaw),
                scale=(self.object_scale, self.object_scale, self.object_scale),
            )
            align_delta = self.renderer.align_prim_bottom_to_z(prim_path, z)
            placements.append(
                {
                    "prim_path": prim_path,
                    "translate": [x, y, z],
                    "rotate_deg": [0.0, 0.0, yaw],
                    "scale": self.object_scale,
                    "bottom_alignment_dz": align_delta,
                }
            )

        return SceneState(
            robot_config_index=config_index,
            robot_joints=authored,
            applied_joints=applied,
            object_placements=placements,
        ).__dict__
