"""Regression checks for shared geometry and the real application contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.paths import ensure_import_paths
ensure_import_paths()
from app.geometry import filter_mirror_detections
from app.projectors_loader import load_projectors
from app.detect import FrameDet
from app.merge import to_global_tracks
from app.local_track import LocalTracklet
from src.contract import Detection
from src.geometry import OfficeModel, Projector, filter_detections_by_mirror, project_detection
from GlobalTracker import GlobalTracker, Tracklet


def bbox_at(point):
    x, y = Projector.get('cam2').plan_to_image(point)
    return [x - 10, y - 50, x + 10, y]


@pytest.mark.parametrize('camera', ['cam2', 'camera_new_2', 'cam_02'])
@pytest.mark.parametrize('kind', ['dict', 'detection', 'frame'])
def test_real_projection_and_detection_contracts(camera, kind):
    def make(point):
        row = dict(camera_id=camera, frame_idx=1, timestamp=1., bbox=bbox_at(point),
                   score=.9, usable_for_reid=True, color_top=[0., 0., 0.], color_bottom=[0., 0., 0.])
        return row if kind == 'dict' else (Detection(**row) if kind == 'detection' else FrameDet(**row))
    mirror, behind, real = make((4.86, 0)), make((4.86, -1)), make((2., 3.))
    kept = filter_mirror_detections([mirror, behind, real])
    assert len(kept) == 1 and kept[0] is real
    assert np.allclose(project_detection(real), (2., 3.))
    assert filter_mirror_detections([mirror], {'enabled': False}) == [mirror]


def test_unknown_camera_and_failed_projection_are_kept():
    unknown = {'camera_id': 'unknown', 'bbox': [0, 0, 10, 10]}
    assert project_detection(unknown) is None
    assert filter_mirror_detections([unknown]) == [unknown]
    det = {'camera_id': 'cam2', 'bbox': [0, 0, 10, 10]}
    assert filter_detections_by_mirror([det], OfficeModel(), lambda d: None) == [det]


def test_floor_polygon_and_other_cameras():
    office = OfficeModel()
    assert office.contains_floor_point((2., 3.))
    assert not office.contains_floor_point((2., 8.))  # outside non-convex floor
    assert not office.contains_floor_point((4.86, -1))
    det = {'camera_id': 'camera_new_1', 'bbox': bbox_at((4.86, -1))}
    assert filter_detections_by_mirror([det], office, lambda d: (4.86, -1)) == [det]


@pytest.mark.parametrize('kind', ['dict', 'list', 'object'])
def test_tracker_projects_observation_formats(kind):
    bbox = bbox_at((2., 3.))
    box = {'bbox': bbox} if kind == 'dict' else (bbox if kind == 'list' else SimpleNamespace(bbox=bbox))
    seg = Tracklet('camera_new_2', 1, 0., 1., np.ones(3), np.zeros(3), np.zeros(3), boxes=[box])
    tracker = GlobalTracker(projectors=load_projectors(['camera_new_2']))
    assert np.allclose(tracker._seg_plan_pos(seg), (2., 3.))
    assert tracker.track([seg])


def test_merge_receives_projectors(monkeypatch):
    import app.merge as merge
    captured = {}
    class SpyTracker(GlobalTracker):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)
    monkeypatch.setattr(merge, 'load_global_tracker_cls', lambda: (SpyTracker, Tracklet, None))
    tl = LocalTracklet('camera_new_2', 1, 0., 1., np.ones(3), np.zeros(3), np.zeros(3),
                      boxes=[{'bbox': bbox_at((2., 3.)), 'frame_idx': 1}])
    assert to_global_tracks([tl])
    assert 'camera_new_2' in captured['projectors']
    assert to_global_tracks([tl], merge_cfg={'bev': {'enabled': False}})
    assert captured['projectors'] == {}


def test_bev_rejects_impossible_transition_with_box_records():
    class IdentityProjector:
        def bbox_bottom_center_to_plan(self, bbox):
            return ((bbox[0] + bbox[2]) / 2, bbox[3])
    tracker = GlobalTracker(projectors={'a': IdentityProjector(), 'b': IdentityProjector()})
    def segment(cam, start, end, x):
        return Tracklet(cam, 1, start, end, np.ones(3), np.zeros(3), np.zeros(3),
                        boxes=[{'bbox': [x, 0, x, 0]}])
    assert tracker._similarity(segment('a', 0, 1, 0), segment('b', 2, 3, 100)) == 0
    assert tracker._similarity(segment('a', 0, 1, 0), segment('b', 2, 3, .5)) > 0


def test_application_filters_before_mot(monkeypatch, tmp_path):
    import app.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    (tmp_path / 'weights.pt').touch()
    monkeypatch.setattr(pipeline, 'discover_videos', lambda *args: [('camera_new_2', tmp_path / 'video.mp4')])
    monkeypatch.setattr(pipeline, '_probe_fps', lambda *args: 25)
    monkeypatch.setattr(pipeline, 'load_yolo_seg', lambda *args: (object(), 'cpu'))
    monkeypatch.setattr(pipeline, 'build_solider_from_cfg', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, 'iter_video_sampled', lambda *args, **kwargs: [(1, 0., None)])
    def det(point):
        return FrameDet('camera_new_2', 1, 0., bbox_at(point), .9, [0,0,0], [0,0,0], True)
    reflection, person = det((4.86, -1)), det((2., 3.))
    monkeypatch.setattr(pipeline, 'detect_people', lambda *args, **kwargs: [reflection, person])
    received = []
    def mot(dets, **kwargs):
        received.extend(dets)
        return []
    monkeypatch.setattr(pipeline, 'run_mot', mot)
    cfg = dict(video_dir='.', model='weights.pt', start_sec=0, max_sec=1, stride=1,
               confidence=.5, iou=.7, imgsz=640, export_mp4=False)
    result = pipeline.run(cfg)
    assert len(received) == 1 and received[0] is person
    assert result['cameras']['camera_new_2']['n_mirror_removed'] == 1
