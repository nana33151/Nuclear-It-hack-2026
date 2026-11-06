"""SOLIDER-REID wrapper around app.reid_solider (vendored tinyvision + Market weights)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.contract import Detection


class SoliderEmbedder:
    def __init__(
        self,
        weights: str | Path,
        only_usable_for_reid: bool = True,
        device: str | None = None,
        variant: str = "swin_small",
        semantic_weight: float = 0.2,
    ) -> None:
        self.weights = Path(weights)
        self.only_usable_for_reid = only_usable_for_reid
        self.device = device
        self.variant = variant
        self.semantic_weight = semantic_weight
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        # Import from repo-root app package (mtmc-person2 is nested).
        import sys

        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from app.reid_solider import SoliderReID

        w = self.weights
        if not w.is_absolute():
            # Prefer repo-root weights/, then local mtmc-person2/weights/
            cand = root / w
            if not cand.exists():
                cand = Path(__file__).resolve().parents[2] / w
            w = cand
        self._model = SoliderReID(
            w,
            variant=self.variant,
            device=self.device or "auto",
            semantic_weight=self.semantic_weight,
        )

    def embed(self, detection: Detection) -> list[float] | None:
        if self.only_usable_for_reid and not detection.usable_for_reid:
            return None
        self._lazy_init()
        crop = None
        if detection.crop_path:
            crop = cv2.imread(str(detection.crop_path))
        if crop is None:
            return None
        emb = self._model.embed_bgr_crop(crop)
        if emb is None:
            return None
        return emb.astype(np.float64).tolist()
