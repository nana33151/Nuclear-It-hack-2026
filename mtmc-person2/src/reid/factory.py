from __future__ import annotations

from typing import Any

from src.reid.color_mock import ColorMockEmbedder
from src.reid.solider import SoliderEmbedder


def build_embedder(backend: str, **kwargs: Any):
    backend = (backend or "none").lower()
    if backend in ("none", "off", "false"):
        return None
    if backend in ("color_mock", "color", "mock"):
        return ColorMockEmbedder(
            only_usable_for_reid=bool(kwargs.get("only_usable_for_reid", True)),
        )
    if backend == "solider":
        return SoliderEmbedder(
            weights=kwargs.get("weights", "weights/swin_s_market.pth"),
            only_usable_for_reid=bool(kwargs.get("only_usable_for_reid", True)),
            device=kwargs.get("device"),
            variant=str(kwargs.get("variant", "swin_small")),
            semantic_weight=float(kwargs.get("semantic_weight", 0.2)),
        )
    raise ValueError(f"Unknown reid.backend: {backend}")
