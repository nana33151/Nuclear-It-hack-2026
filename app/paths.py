"""Repo-root path helpers and optional imports from person folders."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSON1 = ROOT / "person1"
PERSON2 = ROOT / "mtmc-person2"
PERSON3 = ROOT / "person3"


def ensure_import_paths() -> None:
    for path in (ROOT, PERSON1, PERSON2, PERSON3):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def load_global_tracker_cls():
    ensure_import_paths()
    from GlobalTracker import GlobalTracker, Tracklet, GlobalTrack  # type: ignore

    return GlobalTracker, Tracklet, GlobalTrack


def load_bytetrack_cls():
    ensure_import_paths()
    from src.mot.byte_tracker_core import BYTETracker  # type: ignore

    return BYTETracker
