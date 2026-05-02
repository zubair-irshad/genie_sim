from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneTemplate:
    """Validated tabletop layout spec used before paired rendering.

    Genie Sim's LLM scene generator can be wired in later as a producer of this
    same schema. The renderer consumes only validated numeric regions and asset
    query hints, so generated scenes cannot silently place the robot through the
    table or spawn objects outside the reachable tabletop region.
    """

    scene_id: str
    description: str
    table_height: float
    table_size: tuple[float, float]
    robot_mount_xyz: tuple[float, float, float]
    robot_yaw_deg: float
    object_region_xy: tuple[float, float, float, float]
    object_count: tuple[int, int]
    object_queries: tuple[str, ...]
    object_size_range: tuple[float, float]
    hdri_query: str
    background_query: str = ""


DEFAULT_TEMPLATE_SPECS = [
    ("retail_checkout", "retail checkout tabletop with packaged goods", ("bottle", "can", "box", "snack", "cup"), "supermarket,retail,store,indoor"),
    ("grocery_sort", "grocery sorting table", ("fruit", "bottle", "can", "carton", "package"), "supermarket,grocery,indoor"),
    ("home_kitchen", "home kitchen counter manipulation", ("cup", "bowl", "mug", "bottle", "utensil"), "kitchen,home,room,indoor"),
    ("office_desk", "office desk with small tools", ("box", "mouse", "cup", "phone", "book"), "office,room,studio,indoor"),
    ("lab_bench", "laboratory bench with containers", ("bottle", "tube", "box", "cup", "container"), "laboratory,studio,indoor"),
    ("catering_prep", "catering prep table", ("bowl", "plate", "cup", "bottle", "food"), "catering,kitchen,indoor"),
    ("warehouse_pack", "warehouse packing station", ("box", "package", "bottle", "can", "tool"), "warehouse,industrial,indoor"),
    ("home_bedroom_table", "home side table scene", ("book", "cup", "bottle", "phone", "remote"), "home,room,studio,indoor"),
    ("industrial_parts", "industrial small parts workbench", ("tool", "box", "block", "part", "container"), "industrial,warehouse,studio"),
    ("medicine_sort", "medicine and toiletry tabletop", ("medicine", "bottle", "box", "tube", "container"), "clinic,laboratory,indoor"),
    ("coffee_bar", "coffee bar service counter", ("cup", "mug", "bottle", "box", "can"), "cafe,catering,kitchen,indoor"),
    ("clean_tabletop", "minimal studio tabletop", ("block", "box", "bottle", "cup", "can"), "studio,indoor,room"),
]


def generate_default_templates(count: int = 12, seed: int = 42) -> list[SceneTemplate]:
    rng = random.Random(seed)
    templates: list[SceneTemplate] = []
    for idx in range(max(1, count)):
        scene_id, description, queries, hdri_query = DEFAULT_TEMPLATE_SPECS[idx % len(DEFAULT_TEMPLATE_SPECS)]
        table_height = rng.uniform(0.70, 0.82)
        table_x = rng.uniform(1.25, 1.75)
        table_y = rng.uniform(0.78, 1.10)
        margin = 0.18
        region = (-table_x * 0.38, table_x * 0.25, -table_y * 0.32, table_y * 0.32)
        mount_side = rng.choice([-1.0, 1.0])
        robot_x = mount_side * rng.uniform(0.72, 0.98)
        robot_y = rng.uniform(-0.10, 0.10)
        yaw = 180.0 if robot_x > 0 else 0.0
        templates.append(
            SceneTemplate(
                scene_id=f"{idx:02d}_{scene_id}",
                description=description,
                table_height=table_height,
                table_size=(table_x, table_y),
                robot_mount_xyz=(robot_x, robot_y, table_height + 0.002),
                robot_yaw_deg=yaw,
                object_region_xy=(
                    max(region[0], -table_x * 0.5 + margin),
                    min(region[1], table_x * 0.5 - margin),
                    max(region[2], -table_y * 0.5 + margin),
                    min(region[3], table_y * 0.5 - margin),
                ),
                object_count=(2, 4),
                object_queries=tuple(queries),
                object_size_range=(0.075, 0.18),
                hdri_query=hdri_query,
                background_query=description,
            )
        )
    validate_templates(templates)
    return templates


def validate_templates(templates: list[SceneTemplate]) -> None:
    if not templates:
        raise ValueError("At least one scene template is required.")
    for template in templates:
        if not 0.55 <= template.table_height <= 1.10:
            raise ValueError(f"{template.scene_id}: table_height out of range")
        table_x, table_y = template.table_size
        if table_x <= 0.5 or table_y <= 0.4:
            raise ValueError(f"{template.scene_id}: table_size too small")
        x0, x1, y0, y1 = template.object_region_xy
        if not x0 < x1 or not y0 < y1:
            raise ValueError(f"{template.scene_id}: invalid object_region_xy")
        if x0 < -table_x * 0.5 or x1 > table_x * 0.5 or y0 < -table_y * 0.5 or y1 > table_y * 0.5:
            raise ValueError(f"{template.scene_id}: object region exceeds tabletop")
        lo, hi = template.object_count
        if lo < 1 or hi < lo:
            raise ValueError(f"{template.scene_id}: invalid object_count")
        s_lo, s_hi = template.object_size_range
        if s_lo <= 0.0 or s_hi < s_lo:
            raise ValueError(f"{template.scene_id}: invalid object_size_range")


def save_templates(templates: list[SceneTemplate], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(template) for template in templates], indent=2))
    return output


def load_templates(path: str | Path) -> list[SceneTemplate]:
    payload = json.loads(Path(path).read_text())
    templates = [
        SceneTemplate(
            scene_id=item["scene_id"],
            description=item["description"],
            table_height=float(item["table_height"]),
            table_size=tuple(item["table_size"]),
            robot_mount_xyz=tuple(item["robot_mount_xyz"]),
            robot_yaw_deg=float(item["robot_yaw_deg"]),
            object_region_xy=tuple(item["object_region_xy"]),
            object_count=tuple(item["object_count"]),
            object_queries=tuple(item["object_queries"]),
            object_size_range=tuple(item["object_size_range"]),
            hdri_query=item["hdri_query"],
            background_query=item.get("background_query", ""),
        )
        for item in payload
    ]
    validate_templates(templates)
    return templates
