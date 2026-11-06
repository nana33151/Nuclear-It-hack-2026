"""SOLIDER-REID inference wrapper (Swin Market checkpoint).

Uses vendored tinyvision/SOLIDER-REID under third_party/ + Market fine-tuned weights.
Falls back is handled by the caller (color ReID) when weights/backend are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from app.paths import ROOT

SOLIDER_ROOT = ROOT / "third_party" / "SOLIDER-REID"

# Market1501 identity count used when building the classifier head.
_MARKET_NUM_CLASS = 751

_VARIANT_CFG = {
    "swin_small": "configs/market/swin_small.yml",
    "swin_base": "configs/market/swin_base.yml",
    "swin_tiny": "configs/market/swin_tiny.yml",
}


def _ensure_solider_on_path() -> None:
    import sys

    root = str(SOLIDER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested not in ("auto",):
        # Ultralytics-style: "0" / "1" → cuda:N
        if str(requested).isdigit():
            return torch.device(f"cuda:{requested}")
        return torch.device(str(requested))
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SoliderReID:
    """Batch-friendly crop → L2 embedding."""

    def __init__(
        self,
        weights: str | Path,
        *,
        variant: str = "swin_small",
        device: str | None = "auto",
        semantic_weight: float = 0.2,
        image_size: tuple[int, int] = (384, 128),  # H, W
    ) -> None:
        weights_path = Path(weights)
        if not weights_path.is_absolute():
            weights_path = ROOT / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"SOLIDER weights not found: {weights_path}")
        if not SOLIDER_ROOT.exists():
            raise FileNotFoundError(f"Vendored SOLIDER-REID missing: {SOLIDER_ROOT}")

        _ensure_solider_on_path()
        from config import cfg  # type: ignore
        from model.make_model import make_model  # type: ignore

        variant = variant.lower().replace("-", "_")
        if variant not in _VARIANT_CFG:
            raise ValueError(f"Unknown SOLIDER variant={variant}, expected one of {list(_VARIANT_CFG)}")
        yml = SOLIDER_ROOT / _VARIANT_CFG[variant]
        if not yml.exists():
            raise FileNotFoundError(f"SOLIDER config missing: {yml}")

        cfg.defrost()
        cfg.merge_from_file(str(yml))
        cfg.MODEL.PRETRAIN_CHOICE = "self"
        cfg.MODEL.PRETRAIN_PATH = ""
        cfg.MODEL.DEVICE = "cuda" if str(resolve_device(device)).startswith("cuda") else "cpu"
        cfg.freeze()

        self.device = resolve_device(device)
        self.model = make_model(
            cfg,
            num_class=_MARKET_NUM_CLASS,
            camera_num=0,
            view_num=0,
            semantic_weight=float(semantic_weight),
        )
        self.model.load_param(str(weights_path))
        self.model.to(self.device)
        self.model.eval()

        h, w = image_size
        self.transform = T.Compose(
            [
                T.Resize((h, w)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.inference_mode()
    def embed_bgr_crops(self, crops_bgr: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        """Embed OpenCV BGR crops; returns L2-normalized float64 vectors (or None)."""
        if not crops_bgr:
            return []
        tensors = []
        valid_idx: list[int] = []
        out: list[np.ndarray | None] = [None] * len(crops_bgr)
        for i, crop in enumerate(crops_bgr):
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            if crop.ndim != 3 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.transform(pil))
            valid_idx.append(i)
        if not tensors:
            return out
        batch = torch.stack(tensors, dim=0).to(self.device)
        feat, _ = self.model(batch)
        feat = F.normalize(feat, dim=1)
        arr = feat.detach().cpu().numpy().astype(np.float64)
        for j, i in enumerate(valid_idx):
            out[i] = arr[j]
        return out

    def embed_bgr_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        return self.embed_bgr_crops([crop_bgr])[0]


def build_solider_from_cfg(reid_cfg: dict | None, device: str | None = None) -> SoliderReID | None:
    """Return SoliderReID when backend=solider and weights exist; else None."""
    cfg = reid_cfg or {}
    if not cfg.get("enabled", True):
        return None
    backend = str(cfg.get("backend", "color")).lower()
    if backend not in ("solider", "solider_reid"):
        return None
    weights = cfg.get("weights", "weights/swin_s_market.pth")
    try:
        return SoliderReID(
            weights,
            variant=str(cfg.get("variant", "swin_small")),
            device=device or cfg.get("device", "auto"),
            semantic_weight=float(cfg.get("semantic_weight", 0.2)),
        )
    except Exception as e:
        print(f"[reid] SOLIDER unavailable ({e}); falling back to color ReID")
        return None
