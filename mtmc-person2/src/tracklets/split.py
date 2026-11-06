"""Split dirty tracklets by ReID embedding dispersion (Person 2 ownership)."""

from __future__ import annotations

from src.contract import Tracklet


def split_by_reid_dispersion(
    tracklets: list[Tracklet],
    threshold: float,
    per_frame_embs: dict[str, list[list[float] | None]] | None = None,
) -> list[Tracklet]:
    """
    Placeholder-safe splitter.

    Without per-frame embeddings, tracklets are returned unchanged.
    When Person 2 wires SOLIDER, pass per_frame_embs[local_id] aligned to boxes
    and high window-to-window cosine distance will cut the tracklet.
    """
    if not per_frame_embs or threshold <= 0:
        return tracklets

    result: list[Tracklet] = []
    for tr in tracklets:
        embs = per_frame_embs.get(tr.local_id)
        if not embs or len(embs) != len(tr.boxes):
            result.append(tr)
            continue
        cuts = _cut_indices(embs, threshold)
        if not cuts:
            tr.quality = {**tr.quality, "reid_dispersion": _max_disp(embs)}
            result.append(tr)
            continue
        result.extend(_slice_tracklet(tr, cuts, embs))
    return result


def _cosine_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, 1.0 - dot)


def _max_disp(embs: list[list[float] | None]) -> float | None:
    valid = [e for e in embs if e]
    if len(valid) < 2:
        return None
    worst = 0.0
    for i in range(len(valid) - 1):
        worst = max(worst, _cosine_dist(valid[i], valid[i + 1]))
    return worst


def _cut_indices(embs: list[list[float] | None], threshold: float) -> list[int]:
    """Return box indices where a new tracklet should start."""
    cuts: list[int] = []
    last_valid: list[float] | None = None
    for i, emb in enumerate(embs):
        if emb is None:
            continue
        if last_valid is not None and _cosine_dist(last_valid, emb) > threshold and i > 0:
            cuts.append(i)
        last_valid = emb
    return cuts


def _slice_tracklet(tr: Tracklet, cuts: list[int], embs: list[list[float] | None]) -> list[Tracklet]:
    bounds = [0, *cuts, len(tr.boxes)]
    pieces: list[Tracklet] = []
    for pi, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        if e - s <= 0:
            continue
        boxes = tr.boxes[s:e]
        piece_embs = [e for e in embs[s:e] if e]
        mean = None
        if piece_embs:
            dim = len(piece_embs[0])
            acc = [0.0] * dim
            for emb in piece_embs:
                for i, v in enumerate(emb):
                    acc[i] += v
            mean = [v / len(piece_embs) for v in acc]
            n = sum(x * x for x in mean) ** 0.5
            if n > 1e-12:
                mean = [x / n for x in mean]

        pieces.append(
            Tracklet(
                camera_id=tr.camera_id,
                local_id=f"{tr.local_id}_s{pi}",
                t_start=boxes[0].timestamp,
                t_end=boxes[-1].timestamp,
                boxes=boxes,
                reid_emb=mean if mean else tr.reid_emb,
                color_top=tr.color_top,
                color_bottom=tr.color_bottom,
                representative_frame=boxes[len(boxes) // 2].frame_idx,
                split_reason="reid_dispersion" if pi > 0 or len(cuts) else None,
                quality={
                    "n_frames": len(boxes),
                    "n_reid_crops": len(piece_embs),
                    "reid_dispersion": _max_disp(embs[s:e]),
                    "parent_local_id": tr.local_id,
                },
            )
        )
    return pieces
