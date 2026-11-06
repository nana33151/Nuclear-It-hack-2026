"""Deterministic stand-in ReID from Person 1 color histograms.

Unblocks tracklet vectors + dispersion split until SOLIDER weights/torch are ready.
Replace by setting reid.backend: solider once weights exist.
"""

from __future__ import annotations

import math

from src.contract import Detection


class ColorMockEmbedder:
    def __init__(self, only_usable_for_reid: bool = True, dim: int = 32) -> None:
        self.only_usable_for_reid = only_usable_for_reid
        self.dim = dim

    def embed(self, detection: Detection) -> list[float] | None:
        if self.only_usable_for_reid and not detection.usable_for_reid:
            return None
        top = detection.color_top or [0.0, 0.0, 0.0]
        bottom = detection.color_bottom or [0.0, 0.0, 0.0]
        # Pad / project color into fixed dim so cosine math matches SOLIDER path.
        base = list(top) + list(bottom)
        while len(base) < self.dim:
            # mild nonlinear features for stability
            i = len(base)
            base.append(math.sin(sum(base) + i) * 0.01)
        vec = base[: self.dim]
        return _l2(vec)


def _l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]
