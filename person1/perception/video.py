"""Decode in presentation order and retain original frame indices and timestamps."""

from dataclasses import dataclass
from pathlib import Path
import math
import re

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v"}


def discover_videos(input_dir, camera_ids=None):
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Video folder does not exist: {root}")
    videos = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file())
    found = {}
    for path in videos:
        camera_id = path.stem
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", camera_id):
            raise ValueError(f"Use a simple camera filename (letters, numbers, _, -): {path.name}")
        if camera_id in found:
            raise ValueError(f"Duplicate camera ID from filenames: {camera_id}")
        found[camera_id] = path
    if camera_ids:
        missing = set(camera_ids) - set(found)
        if missing:
            raise ValueError(f"Unknown cameras: {sorted(missing)}; available: {list(found)}")
        found = {key: value for key, value in found.items() if key in camera_ids}
    if not found:
        raise ValueError(f"No supported video files in {root}")
    return found


def probe_video(path):
    import av
    path = Path(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else None
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"No valid nominal FPS for tracker handoff: {path}")
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None else
                    container.duration / av.time_base if container.duration is not None else None)
        stat = path.stat()
        first_frame = next(container.decode(stream), None)
        first_pts = (float(first_frame.pts * first_frame.time_base)
                     if first_frame is not None and first_frame.pts is not None and first_frame.time_base is not None else None)
        return {
            "camera_id": path.stem,
            "source_video": str(path.resolve()),
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": fps,
            "duration_seconds": duration,
            "frame_count_expected": int(stream.frames) if stream.frames else None,
            "frame_count_estimate": round(duration * fps) if duration is not None and fps else None,
            "time_base": str(stream.time_base),
            "codec": stream.codec_context.name,
            "source_first_pts_seconds": first_pts,
        }


@dataclass
class VideoFrame:
    frame_idx: int
    timestamp: float
    timestamp_source: str
    image: object


def iter_video_frames(path, offset=0.0, max_frames=None, start_frame=0, timestamp_origin=None):
    import av
    if not math.isfinite(offset):
        raise ValueError("Camera offset must be finite")
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    if timestamp_origin is not None and not math.isfinite(timestamp_origin):
        raise ValueError("timestamp_origin must be finite or None")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else None
        origin = timestamp_origin
        previous = None
        source_mode = None
        for idx, frame in enumerate(container.decode(stream)):
            if max_frames is not None and idx >= start_frame + max_frames:
                break
            has_pts = frame.pts is not None and frame.time_base is not None
            if source_mode is None:
                source_mode = "pts" if has_pts else "fps"
            if source_mode == "pts":
                if not has_pts:
                    raise ValueError(f"Missing PTS after timestamped frames in {path}, frame {idx}")
                pts_time = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = pts_time
                timestamp = pts_time - origin + offset
            else:
                if not fps or fps <= 0:
                    raise ValueError(f"Neither frame timestamps nor valid FPS available: {path}")
                timestamp = idx / fps + offset - (timestamp_origin or 0.0)
            if previous is not None and timestamp <= previous:
                raise ValueError(f"Non-increasing timestamps in {path}, frame {idx}: {timestamp} <= {previous}")
            previous = timestamp
            if idx < start_frame:
                continue
            yield VideoFrame(idx, timestamp, source_mode, frame.to_ndarray(format="bgr24"))
