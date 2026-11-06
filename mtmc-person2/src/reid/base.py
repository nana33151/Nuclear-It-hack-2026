from __future__ import annotations

from typing import Protocol

from src.contract import Detection


class Embedder(Protocol):
    def embed(self, detection: Detection) -> list[float] | None:
        ...
