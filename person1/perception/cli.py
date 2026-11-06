from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve()
    defaults = json.loads((PROJECT_ROOT / "configs/default.json").read_text(encoding="utf-8"))
    supplied = json.loads(path.read_text(encoding="utf-8"))
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    defaults.update(supplied)
    # All configured paths are relative to the project, independent of shell cwd.
    for name in ["input_dir", "model", "schp_model"]:
        value = Path(defaults[name])
        defaults[name] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return defaults


def check_config(config):
    for key in ["confidence", "reid_min_confidence", "iou"]:
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be > 0 and <= 1")
    if not isinstance(config["imgsz"], int) or config["imgsz"] < 32 or config["imgsz"] % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if config["max_frames"] is not None and (not isinstance(config["max_frames"], int) or config["max_frames"] < 1):
        raise ValueError("max_frames must be a positive integer or null")
    if not isinstance(config["preview_frames"], int) or config["preview_frames"] < 0:
        raise ValueError("preview_frames must be a nonnegative integer")
    if type(config["start_frame"]) is not int or config["start_frame"] < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    if config["color_mode"] not in {"schp", "spatial", "none"}:
        raise ValueError("color_mode must be schp, spatial or none")
    if not isinstance(config["color_only_usable"], bool):
        raise ValueError("color_only_usable must be a boolean")
    if not isinstance(config["camera_offsets"], dict):
        raise ValueError("camera_offsets must be an object of camera_id: seconds")
    for camera, offset in config["camera_offsets"].items():
        if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(offset):
            raise ValueError(f"Invalid camera offset: {camera}")
    origin = config["timestamp_origin"]
    if isinstance(origin, bool) or not isinstance(origin, (int, float)) or not math.isfinite(origin):
        raise ValueError("timestamp_origin must be a finite number shared by ALL cameras")


def make_parser():
    parser = argparse.ArgumentParser(description="Person1: silhouettes, crops and features for person2")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect input videos without loading models")
    inspect.add_argument("--input", default=str(PROJECT_ROOT / "Video"))
    run = commands.add_parser("run", help="Process videos into a versioned person2 export")
    run.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.json"))
    run.add_argument("--input")
    run.add_argument("--output")
    run.add_argument("--camera", action="append", help="Filename without extension; repeat to select several")
    run.add_argument("--model")
    run.add_argument("--device", help="auto, cpu, mps, or CUDA device number")
    run.add_argument("--imgsz", type=int)
    run.add_argument("--confidence", type=float)
    run.add_argument("--color-mode", choices=["schp", "spatial", "none"])
    run.add_argument("--max-frames", type=int, help="Limit frames PER camera; omit for whole video")
    run.add_argument("--start-frame", type=int, help="Start at this source index, retaining original timestamps")
    run.add_argument("--preview-frames", type=int)
    validate = commands.add_parser("validate", help="Validate handoff files and image geometry")
    validate.add_argument("output")
    validate.add_argument("--quick", action="store_true", help="Skip decoding every saved crop and mask")
    pack = commands.add_parser("pack", help="Create a portable ZIP with export, reader and person2 instructions")
    pack.add_argument("output")
    pack.add_argument("archive")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            from .video import discover_videos, probe_video
            print(json.dumps([probe_video(path) for path in discover_videos(args.input).values()], ensure_ascii=False, indent=2))
        elif args.command == "validate":
            from .validation import validate_export
            print(json.dumps(validate_export(args.output, verify_images=not args.quick), ensure_ascii=False, indent=2))
        elif args.command == "pack":
            from .handoff import pack_handoff
            print(json.dumps(pack_handoff(args.output, args.archive), ensure_ascii=False, indent=2))
        else:
            from .pipeline import run
            config = load_config(args.config)
            for argument, key in [("input", "input_dir"), ("model", "model"), ("device", "device"),
                                  ("imgsz", "imgsz"), ("confidence", "confidence"),
                                  ("color_mode", "color_mode"), ("max_frames", "max_frames"),
                                  ("start_frame", "start_frame"),
                                  ("preview_frames", "preview_frames")]:
                value = getattr(args, argument)
                if value is not None:
                    config[key] = value
            check_config(config)
            output = args.output or PROJECT_ROOT / "outputs" / datetime.now().strftime("run-%Y%m%d-%H%M%S")
            manifest = run(config, output, args.camera)
            print(json.dumps({"status": manifest["status"], "output": str(Path(output).resolve()),
                              "frames": sum(c["processed_frames"] for c in manifest["cameras"]),
                              "detections": sum(c["detections_count"] for c in manifest["cameras"])}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Incomplete output remains marked failed; start a new output folder.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
