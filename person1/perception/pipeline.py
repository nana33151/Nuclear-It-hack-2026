"""Streaming video -> segmentations, original crops, color and quality features."""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import math
import os
import time

from .video import discover_videos, iter_video_frames, probe_video


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested):
    import torch
    if requested in {"mps", "mps:0"} and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable to this Python process. Choose --device cpu or run from a terminal with GPU access.")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checked_write_image(path, image, parameters=None):
    import cv2
    if not cv2.imwrite(str(path), image, parameters or []):
        raise OSError(f"Could not save image: {path}")


def extract_detections(result, image, camera_id, frame_idx, timestamp, output_root, camera_dir,
                       quality_config, color_extractor, color_only_usable, color_method, diagnostics=None,
                       reid_min_confidence=0.5):
    import cv2
    import numpy as np
    from .features import assess_quality

    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None:
        raise ValueError("Detector returned boxes without silhouettes. Use YOLO11 -seg weights.")
    h, w = image.shape[:2]
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    if len(masks) != len(boxes) or masks.shape[1:] != (h, w):
        raise ValueError(f"Masks must match original frame {w}x{h}; got {masks.shape}. retina_masks=True is required.")
    records = []
    for raw_box, score, cls, full_mask in zip(boxes, scores, classes, masks):
        if int(cls) != 0:
            continue
        x1 = max(0, min(w, math.floor(float(raw_box[0]))))
        y1 = max(0, min(h, math.floor(float(raw_box[1]))))
        x2 = max(0, min(w, math.ceil(float(raw_box[2]))))
        y2 = max(0, min(h, math.ceil(float(raw_box[3]))))
        if x2 <= x1 or y2 <= y1:
            if diagnostics is not None:
                diagnostics["invalid_boxes_skipped"] = diagnostics.get("invalid_boxes_skipped", 0) + 1
            continue
        bbox = [x1, y1, x2, y2]
        crop = image[y1:y2, x1:x2].copy()
        mask = ((full_mask[y1:y2, x1:x2] > .5).astype(np.uint8) * 255)
        if not mask.any():
            if diagnostics is not None:
                diagnostics["empty_masks_skipped"] = diagnostics.get("empty_masks_skipped", 0) + 1
            continue
        n = len(records)
        file_id = f"f{frame_idx:09d}_d{n:03d}"
        detection_id = f"{camera_id}:f{frame_idx:09d}:d{n:03d}"
        crop_path = camera_dir / "crops" / f"{file_id}.jpg"
        mask_path = camera_dir / "masks" / f"{file_id}.png"
        quality = assess_quality(crop, mask, bbox, (h, w), quality_config)
        if score < reid_min_confidence:
            quality["usable_for_reid"] = False
            quality["quality"]["reasons"].append("low_detection_confidence")
        if color_extractor is not None and (quality["usable_for_reid"] or not color_only_usable):
            color = color_extractor.extract(crop, mask)
        else:
            color = {"upper_color_hist": None, "lower_color_hist": None,
                     "color_method": color_method,
                     "color_quality": {"reason": "crop_not_usable" if color_extractor else "disabled"}}
        checked_write_image(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        checked_write_image(mask_path, mask)
        records.append({
            "detection_id": detection_id, "camera_id": camera_id,
            "frame_idx": frame_idx, "timestamp": timestamp,
            "bbox": bbox, "score": float(score), "class_id": 0, "class_name": "person",
            "mask_path": mask_path.relative_to(output_root).as_posix(),
            "crop_path": crop_path.relative_to(output_root).as_posix(),
            "mask_format": "png_crop", "mask_origin": [x1, y1], "mask_size": [x2 - x1, y2 - y1],
            **quality, **color,
        })
    return records


def draw_overlay(image, detections, output_root, frame_idx, timestamp):
    import cv2
    import numpy as np
    display = image.copy()
    for i, record in enumerate(detections):
        x1, y1, x2, y2 = record["bbox"]
        color = (90, 220, 100) if record["usable_for_reid"] else (0, 180, 255)
        mask = cv2.imread(str(output_root / record["mask_path"]), cv2.IMREAD_GRAYSCALE)
        area = display[y1:y2, x1:x2]
        inside = mask > 0
        area[inside] = (area[inside] * .65 + np.asarray(color) * .35).astype(np.uint8)
        cv2.rectangle(display, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        cv2.putText(display, f"det {i} {record['score']:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    cv2.putText(display, f"frame {frame_idx} | {timestamp:.3f}s | detections {len(detections)}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return display


def process_camera(model, info, config, output_root, quality_config, extractor, color_method):
    import cv2

    camera_id = info["camera_id"]
    directory = output_root / "cameras" / camera_id
    (directory / "crops").mkdir(parents=True)
    (directory / "masks").mkdir()
    frames_path = directory / "frames.jsonl"
    detections_path = directory / "detections.jsonl"
    info.update(status="running", frames_path=frames_path.relative_to(output_root).as_posix(),
                detections_path=detections_path.relative_to(output_root).as_posix(),
                processed_frames=0, detections_count=0, usable_count=0)
    info["diagnostics"] = {"invalid_boxes_skipped": 0, "empty_masks_skipped": 0}
    started = time.monotonic()
    writer = None
    snapshot_saved = False
    max_frames = config.get("max_frames")
    offset = config.get("camera_offsets", {}).get(camera_id, 0.0)
    info["timestamp_offset"] = offset
    last_frame = None
    try:
        with frames_path.open("w", encoding="utf-8") as frames_file, detections_path.open("w", encoding="utf-8") as detections_file:
            for frame in iter_video_frames(info["source_video"], offset=offset, max_frames=max_frames,
                                           start_frame=config["start_frame"], timestamp_origin=config["timestamp_origin"]):
                image = frame.image
                h, w = image.shape[:2]
                if (w, h) != (info["width"], info["height"]):
                    raise ValueError(f"Resolution changes inside {camera_id}; frame {frame.frame_idx}")
                result = model.predict(source=image, classes=[0], conf=config["confidence"],
                                       iou=config["iou"], imgsz=config["imgsz"],
                                       device=config["resolved_device"], retina_masks=True,
                                       verbose=False, save=False)[0]
                detections = extract_detections(
                    result, image, camera_id, frame.frame_idx, frame.timestamp, output_root, directory,
                    quality_config, extractor, config["color_only_usable"], color_method, info["diagnostics"],
                    config["reid_min_confidence"])
                for record in detections:
                    detections_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                frames_file.write(json.dumps({
                    "camera_id": camera_id, "frame_idx": frame.frame_idx, "timestamp": frame.timestamp,
                    "timestamp_source": frame.timestamp_source, "width": w, "height": h,
                    "detection_ids": [d["detection_id"] for d in detections],
                }, allow_nan=False) + "\n")
                info["processed_frames"] += 1
                info["detections_count"] += len(detections)
                info["usable_count"] += sum(d["usable_for_reid"] for d in detections)
                last_frame = frame
                selected_index = info["processed_frames"] - 1
                if selected_index < config["preview_frames"] or (detections and not snapshot_saved):
                    overlay = draw_overlay(image, detections, output_root, frame.frame_idx, frame.timestamp)
                    if detections and not snapshot_saved:
                        checked_write_image(directory / "preview.jpg", overlay)
                        info["preview_image"] = (directory / "preview.jpg").relative_to(output_root).as_posix()
                        snapshot_saved = True
                    if selected_index < config["preview_frames"]:
                        if writer is None:
                            preview_path = directory / "preview.mp4"
                            writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"] or 25.0, (w, h))
                            if not writer.isOpened():
                                raise RuntimeError(f"Cannot create preview: {preview_path}")
                            info["preview_video"] = preview_path.relative_to(output_root).as_posix()
                            info["preview_timing"] = "nominal_fps_visual_check_only"
                        writer.write(overlay)
                if info["processed_frames"] % 100 == 0:
                    elapsed = time.monotonic() - started
                    print(f"{camera_id}: {info['processed_frames']} frames, {info['detections_count']} detections, "
                          f"{info['processed_frames'] / elapsed:.2f} frames/s", flush=True)
            if last_frame is None:
                raise ValueError(f"No frames decoded: {info['source_video']}")
        limited = max_frames is not None and info["processed_frames"] >= max_frames
        info["coverage"] = {"start_frame": config["start_frame"], "last_frame": last_frame.frame_idx,
                            "limited_by_max_frames": limited, "whole_video": not limited and config["start_frame"] == 0}
        with frames_path.open(encoding="utf-8") as first_record_file:
            info["first_timestamp"] = json.loads(first_record_file.readline())["timestamp"]
        info["last_timestamp"] = last_frame.timestamp
        info["status"] = "complete"
    except BaseException:
        info["status"] = "failed"
        raise
    finally:
        if writer is not None:
            writer.release()
        info["processing_seconds"] = round(time.monotonic() - started, 3)
    print(f"{camera_id}: complete, {info['processed_frames']} frames, {info['detections_count']} detections", flush=True)


def run(config, output_dir, camera_ids=None):
    # Keep tool caches in this project; source video pixels are never uploaded.
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / ".cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".cache/matplotlib"))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO
    from ultralytics import settings
    settings.update({"sync": False})
    from .features import QualityConfig, SpatialColorExtractor, SCHPColorExtractor

    videos = discover_videos(config["input_dir"], camera_ids)
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output folder is not empty; choose a NEW output folder: {output_root}")
    model_path = Path(config["model"])
    if not model_path.is_file():
        raise ValueError(f"YOLO weights not found: {model_path}. See README model setup.")
    quality_config = QualityConfig(**config["quality"])
    mode = config["color_mode"]
    if mode == "schp":
        extractor = SCHPColorExtractor(config["schp_model"])
        color_method = "schp_atr_hsv"
    elif mode == "spatial":
        extractor = SpatialColorExtractor()
        color_method = "spatial_hsv"
    else:
        extractor = None
        color_method = "none"
    config = dict(config)
    config["resolved_device"] = choose_device(config["device"])
    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError("Only segmentation weights are supported (filename ending -seg.pt).")
    cameras = [probe_video(path) for path in videos.values()]
    for info in cameras:
        info["status"] = "pending"
    output_root.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ["nuclerit-perception", "ultralytics", "torch", "torchvision", "numpy", "av", "opencv-python", "onnxruntime"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema_version": "1.0", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "versions": versions,
        "models": {"yolo": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)}},
        "cameras": cameras,
        "coordinate_system": "original_frame_xyxy_end_exclusive",
        "timestamp_policy": "source_video_pts_minus_shared_timestamp_origin_plus_camera_offset; fps_fallback_only_when_initial_pts_missing",
        "histogram": {"space": "OpenCV HSV", "bins": [16, 8, 8], "length": 32,
                      "normalization": "each_channel_L1_then_concat_divide_3"},
    }
    if mode == "schp":
        manifest["models"]["schp"] = {"path": str(Path(config["schp_model"]).resolve()),
                                       "sha256": sha256_file(config["schp_model"])}
    for name, entry in manifest["models"].items():
        source_path = Path(entry["path"]).with_suffix(Path(entry["path"]).suffix + ".source.json")
        if source_path.is_file():
            source = json.loads(source_path.read_text())
            if source.get("sha256") == entry["sha256"]:
                entry["provenance"] = source
    manifest_path = output_root / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"Output: {output_root}\nDevice: {config['resolved_device']}; color: {mode}; cameras: {len(cameras)}", flush=True)
    try:
        for info in cameras:
            info["status"] = "running"
            atomic_json(manifest_path, manifest)
            process_camera(model, info, config, output_root, quality_config, extractor, color_method)
            atomic_json(manifest_path, manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
    return manifest
