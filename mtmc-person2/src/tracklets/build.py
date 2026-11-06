"""Convert MOT raw tracks into Person 3 tracklet contract."""

from __future__ import annotations

import statistics
from typing import Callable, Sequence

from src.contract import BoxObservation, Detection, Tracklet
from src.mot.base import RawTrack

EmbedFn = Callable[[Detection], list[float] | None]


def raw_tracks_to_tracklets(
    camera_id: str,
    raw_tracks: Sequence[RawTrack],
    embed_fn: EmbedFn | None = None,
    min_frames: int = 3,
    representative: str = "middle",
) -> tuple[list[Tracklet], dict[str, list[list[float] | None]]]:
    out: list[Tracklet] = []
    per_frame: dict[str, list[list[float] | None]] = {}

    for raw in raw_tracks:
        dets = sorted(raw.detections, key=lambda d: d.frame_idx)
        if len(dets) < min_frames:
            continue

        boxes = [
            BoxObservation(
                frame_idx=d.frame_idx,
                timestamp=d.timestamp,
                bbox=list(d.bbox),
                score=d.score,
                usable_for_reid=d.usable_for_reid,
            )
            for d in dets
        ]

        frame_embs: list[list[float] | None] = []
        emb_list: list[list[float]] = []
        if embed_fn is not None:
            for d in dets:
                emb = embed_fn(d)
                if emb is not None:
                    emb = _l2_normalize(emb)
                    emb_list.append(emb)
                    frame_embs.append(emb)
                else:
                    frame_embs.append(None)
        else:
            frame_embs = [None] * len(dets)

        reid_emb = _mean_emb(emb_list) if emb_list else None
        color_top = _median_color([d.color_top for d in dets if d.color_top])
        color_bottom = _median_color([d.color_bottom for d in dets if d.color_bottom])
        rep = _pick_representative(dets, representative)
        local_id = str(raw.local_id)

        out.append(
            Tracklet(
                camera_id=camera_id,
                local_id=local_id,
                t_start=dets[0].timestamp,
                t_end=dets[-1].timestamp,
                boxes=boxes,
                reid_emb=reid_emb,
                color_top=color_top,
                color_bottom=color_bottom,
                representative_frame=rep.frame_idx,
                quality={
                    "n_frames": len(dets),
                    "n_reid_crops": len(emb_list),
                    "reid_dispersion": None,
                },
            )
        )
        per_frame[local_id] = frame_embs

    return out, per_frame


def _pick_representative(dets: list[Detection], mode: str) -> Detection:
    if mode == "best_score":
        return max(dets, key=lambda d: d.score)
    return dets[len(dets) // 2]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 1e-12:
        return list(vec)
    return [x / norm for x in vec]


def _mean_emb(embs: list[list[float]]) -> list[float]:
    dim = len(embs[0])
    acc = [0.0] * dim
    for e in embs:
        for i, v in enumerate(e):
            acc[i] += v
    mean = [v / len(embs) for v in acc]
    return _l2_normalize(mean)


def _median_color(colors: list[list[float] | None]) -> list[float] | None:
    valid = [c for c in colors if c]
    if not valid:
        return None
    dim = len(valid[0])
    return [float(statistics.median(c[i] for c in valid)) for i in range(dim)]
