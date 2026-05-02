from __future__ import annotations

import random
from pathlib import Path

from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.scene_templates.templates import SceneTemplate


class ObjectPoolSampler:
    """Samples USD foreground objects per scene template.

    Assets are selected per generated scene/pair, not once for the whole run.
    This avoids the repeated-bottle failure mode while keeping object choices
    tied to the template's natural-language domain.
    """

    def __init__(self, index: AssetIndex, seed: int = 42):
        self.index = index
        self.rng = random.Random(seed)
        self._all_objects = [path for path in index.objects() if path.suffix.lower() in {".usd", ".usda", ".usdc"}]

    def sample(self, template: SceneTemplate, count: int | None = None) -> list[Path]:
        if not self._all_objects:
            return []
        lo, hi = template.object_count
        requested = count if count is not None else self.rng.randint(lo, hi)
        candidates = self._ranked_candidates(template)
        if len(candidates) < requested:
            candidates = candidates + [path for path in self._all_objects if path not in candidates]
        if len(candidates) <= requested:
            return candidates[:requested]
        head = candidates[: max(requested * 4, requested)]
        return self.rng.sample(head, requested)

    def _ranked_candidates(self, template: SceneTemplate) -> list[Path]:
        ranked = []
        for path in self._all_objects:
            low = str(path).lower()
            score = sum(query.lower() in low for query in template.object_queries)
            ranked.append((score, self.rng.random(), path))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked if item[0] > 0] or [item[2] for item in ranked]
