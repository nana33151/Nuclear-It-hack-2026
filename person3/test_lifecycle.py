"""Smoke tests for GlobalTracker lifecycle (Unconfident / Confident / Lost / Died)."""

from __future__ import annotations

import numpy as np

from GlobalTracker import GlobalTracker, Tracklet, TransitionEdge


def _tl(
    cam: str,
    lid: int,
    t0: float,
    t1: float,
    seed: float,
) -> Tracklet:
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.standard_normal(64)
    v = v / (np.linalg.norm(v) or 1.0)
    c = rng.random(3)
    return Tracklet(
        camera_id=cam,
        local_id=lid,
        t_start=t0,
        t_end=t1,
        reid=v,
        color_upper=c,
        color_lower=c * 0.9,
    )


def _same_person(cam: str, lid: int, t0: float, t1: float) -> Tracklet:
    base = _tl(cam, lid, t0, t1, seed=42.0)
    return base


def test_short_flicker_no_global_id() -> None:
    gt = GlobalTracker(T=8.0, min_sim=0.5, min_hits=2, min_confirm_sec=0.4)
    # one very short segment -> should not appear in output
    tl = _tl("cam_a", 1, 0.0, 0.15, seed=1.0)
    tracks = gt.track([tl])
    assert tracks == [], f"expected no tracks, got {tracks}"


def test_lost_rejoin_same_id() -> None:
    gt = GlobalTracker(T=8.0, min_sim=0.55, min_hits=1, min_confirm_sec=0.1)
    a = _same_person("cam_a", 1, 0.0, 2.0)
    b = _same_person("cam_b", 2, 4.0, 6.0)  # gap 2s < T
    tracks = gt.track([a, b])
    assert len(tracks) == 1, tracks
    assert len(tracks[0].segments) == 2
    assert tracks[0].close_reason in ("end_of_window", "gap_exceeded")


def test_gap_exceeds_T_new_id() -> None:
    gt = GlobalTracker(
        T=3.0,
        min_sim=0.55,
        min_hits=1,
        min_confirm_sec=0.1,
        max_transition_time=3.0,
    )
    a = _same_person("cam_a", 1, 0.0, 2.0)
    b = _same_person("cam_b", 2, 10.0, 12.0)  # gap 8s > T and max_transition_time
    tracks = gt.track([a, b])
    assert len(tracks) == 2, tracks
    assert all(tr.close_reason in ("gap_exceeded", "end_of_window") for tr in tracks)


def test_transition_window_outlives_T() -> None:
    edges = {("cam_a", "cam_b"): TransitionEdge(0.0, 12.0, 1.0)}
    gt = GlobalTracker(
        T=3.0,
        min_sim=0.5,
        min_hits=1,
        min_confirm_sec=0.1,
        transitions=edges,
    )
    a = _same_person("cam_a", 1, 0.0, 2.0)
    b = _same_person("cam_b", 2, 10.0, 12.0)  # gap 8s > T, < edge.max_time
    tracks = gt.track([a, b])
    assert len(tracks) == 1, tracks
    assert len(tracks[0].segments) == 2


def test_low_edge_weight_does_not_block_min_sim() -> None:
    edges = {("cam_a", "cam_b"): TransitionEdge(0.0, 10.0, 0.35)}
    gt = GlobalTracker(
        T=8.0,
        min_sim=0.55,
        min_hits=1,
        min_confirm_sec=0.1,
        transitions=edges,
    )
    a = _same_person("cam_a", 1, 0.0, 2.0)
    b = _same_person("cam_b", 2, 2.0, 4.0)
    tracks = gt.track([a, b])
    assert len(tracks) == 1, tracks


def test_overlap_rejected_when_min_time_positive() -> None:
    edges = {("cam_a", "cam_b"): TransitionEdge(min_time=4.0, max_time=10.0, weight=1.0)}
    gt = GlobalTracker(
        T=8.0,
        min_sim=0.5,
        min_hits=1,
        min_confirm_sec=0.1,
        transitions=edges,
    )
    a = _same_person("cam_a", 1, 0.0, 5.0)
    b = _same_person("cam_b", 2, 4.0, 8.0)  # overlap, min_time=4
    tracks = gt.track([a, b])
    assert len(tracks) == 2, tracks


def test_unconfident_fragments_merge() -> None:
    gt = GlobalTracker(T=8.0, min_sim=0.5, min_hits=2, min_confirm_sec=10.0)
    a = _same_person("cam_a", 1, 0.0, 0.2)
    b = _same_person("cam_a", 2, 0.5, 2.0)
    tracks = gt.track([a, b])
    assert len(tracks) == 1, tracks
    assert len(tracks[0].segments) == 2


def test_transition_missing_edge_no_merge() -> None:
    edges = {
        ("cam_a", "cam_b"): TransitionEdge(0.0, 10.0, 1.0),
    }
    gt = GlobalTracker(
        T=8.0,
        min_sim=0.5,
        min_hits=1,
        min_confirm_sec=0.1,
        transitions=edges,
    )
    a = _same_person("cam_a", 1, 0.0, 2.0)
    b = _same_person("cam_c", 2, 3.0, 5.0)
    tracks = gt.track([a, b])
    assert len(tracks) == 2, tracks


def test_transition_gap_outside_window() -> None:
    edges = {
        ("cam_a", "cam_b"): TransitionEdge(min_time=5.0, max_time=10.0, weight=1.0),
    }
    gt = GlobalTracker(
        T=8.0,
        min_sim=0.5,
        min_hits=1,
        min_confirm_sec=0.1,
        transitions=edges,
    )
    a = _same_person("cam_a", 1, 0.0, 2.0)
    b = _same_person("cam_b", 2, 5.0, 7.0)  # gap 3s < min_time 5
    tracks = gt.track([a, b])
    assert len(tracks) == 2, tracks


def test_tracks_json_shape() -> None:
    gt = GlobalTracker()
    a = _same_person("cam_a", 1, 0.0, 5.0)
    tracks = gt.track([a])
    assert len(tracks) == 1
    tr = tracks[0]
    for key in ("track_id", "t_start", "t_end", "cameras", "close_reason", "segments"):
        assert hasattr(tr, key), key
    seg = tr.segments[0]
    for key in ("camera_id", "local_id", "t_start", "t_end"):
        assert key in seg, seg


if __name__ == "__main__":
    test_short_flicker_no_global_id()
    test_lost_rejoin_same_id()
    test_gap_exceeds_T_new_id()
    test_transition_window_outlives_T()
    test_low_edge_weight_does_not_block_min_sim()
    test_overlap_rejected_when_min_time_positive()
    test_unconfident_fragments_merge()
    test_transition_missing_edge_no_merge()
    test_transition_gap_outside_window()
    test_tracks_json_shape()
    print("ok: lifecycle smoke tests passed")
