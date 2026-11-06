#!/usr/bin/env python3
"""Run Person 2 tracklet pipeline.

Examples:
  py -3 scripts/run_tracklets.py
  py -3 scripts/run_tracklets.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Person 2: detections → local tracklets")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    summary = run(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
