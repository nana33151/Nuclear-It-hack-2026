"""Run from the project root: python -m examples.read_for_person2 OUTPUT_DIR."""

from __future__ import annotations

import argparse
import json

from perception.reader import PerceptionReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить подключение person2 к экспорту person1")
    parser.add_argument("output_dir", help="Папка с manifest.json")
    parser.add_argument("--camera", help="Одна камера; по умолчанию все")
    parser.add_argument("--limit", type=int, default=5, help="Показать первые N кадров каждой камеры")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit должен быть положительным")
    reader = PerceptionReader(args.output_dir)
    for camera_id in ([args.camera] if args.camera else reader.camera_ids):
        # Instantiate one local tracker per camera HERE, outside the frame loop.
        for number, frame in enumerate(reader.iter_frames(camera_id)):
            boxes = frame.boxes  # float32 (N,5), including low-confidence boxes.
            # The upstream ByteTrack API can be connected here:
            # tracks = tracker.update(boxes, (frame.height, frame.width),
            #                         (frame.height, frame.width))
            # All boxes are ALREADY in original pixels, so rescaling is 1.
            # Other ByteTrack/BoT-SORT packages may use a different signature.
            # Call the tracker even when boxes.shape == (0, 5).

            reid_candidates = [det for det in frame.detections if det["usable_for_reid"]]
            # ReID should only use suitable candidates. Loading one crop:
            # rgb = reader.read_crop(reid_candidates[0])
            # mask = reader.read_mask(reid_candidates[0])
            # Match tracker results back to detection_id using the association
            # returned by your tracker or one-to-one geometric matching.
            # Never zip(track_ids, detections): ordering/length need not match.
            print(json.dumps({
                "camera_id": frame.camera_id,
                "frame_idx": frame.frame_idx,
                "timestamp": frame.timestamp,
                "boxes_shape": list(boxes.shape),
                "detection_ids": list(frame.detection_ids),
                "reid_candidates": len(reid_candidates),
            }, ensure_ascii=False))
            if number + 1 >= args.limit:
                break


if __name__ == "__main__":
    main()
