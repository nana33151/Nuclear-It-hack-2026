"""SOLIDER-REID wrapper. Requires torch + weights file.

Until then pipeline should use reid.backend: color_mock.
"""

from __future__ import annotations

from pathlib import Path

from src.contract import Detection


class SoliderEmbedder:
    def __init__(
        self,
        weights: str | Path,
        only_usable_for_reid: bool = True,
        device: str | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.only_usable_for_reid = only_usable_for_reid
        self.device = device
        self._model = None
        self._torch = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "SOLIDER needs torch. Use a Python 3.10–3.11 env and pip install torch, "
                "or set reid.backend: color_mock until then."
            ) from e
        if not self.weights.exists():
            raise FileNotFoundError(
                f"SOLIDER weights not found: {self.weights}. "
                "Download Swin-B ReID weights into weights/ and update configs/default.yaml."
            )
        self._torch = torch
        # Full SOLIDER-REID graph is vendored later; keep explicit failure until wired.
        raise NotImplementedError(
            "SOLIDER model graph not vendored yet. Weights path is reserved; "
            "use reid.backend: color_mock for now, then drop in tinyvision/SOLIDER-REID inference."
        )

    def embed(self, detection: Detection) -> list[float] | None:
        if self.only_usable_for_reid and not detection.usable_for_reid:
            return None
        if not detection.crop_path:
            return None
        self._lazy_init()
        return None
