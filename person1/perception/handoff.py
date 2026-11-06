"""Build a portable person2 bundle after checking the export is complete."""

from pathlib import Path
import json
import zipfile

from .reader import PerceptionReader
from .validation import validate_export


def pack_handoff(output_dir, archive_path):
    root = Path(output_dir).resolve()
    archive = Path(archive_path).resolve()
    if archive.exists():
        raise ValueError(f"Archive already exists: {archive}")
    if archive.is_relative_to(root):
        raise ValueError("Put the ZIP outside the export folder")
    report = validate_export(root, verify_images=True)
    reader = PerceptionReader(root)
    project = Path(__file__).resolve().parents[1]
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".partial")
    if temporary.exists():
        raise ValueError(f"Unfinished archive exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
            for name in ["perception/__init__.py", "perception/reader.py", "perception/validation.py",
                         "examples/__init__.py", "examples/read_for_person2.py", "docs/PERSON2_CONTRACT.md"]:
                bundle.write(project / name, name)
            bundle.writestr("requirements.txt", "numpy>=1.26,<3\nPillow>=10,<13\n")
            bundle.writestr("README.txt", "Person1 -> person2\n\n"
                            "1. Install: python -m pip install -r requirements.txt\n"
                            "2. Read: python -m examples.read_for_person2 data --limit 5\n"
                            "3. Integration specification: docs/PERSON2_CONTRACT.md\n\n"
                            "YOLO, SCHP, torch and original videos are not needed to read this export.\n"
                            "Check camera coverage in data/manifest.json: this can be a sample.\n")
            bundle.writestr("validation.json", json.dumps(report, ensure_ascii=False, indent=2))
            bundle.write(root / "manifest.json", "data/manifest.json")
            for camera_id, info in reader.cameras.items():
                for field in ["frames_path", "detections_path", "preview_image", "preview_video"]:
                    if info.get(field):
                        path = reader.asset_path(info[field])
                        bundle.write(path, "data/" + path.relative_to(root).as_posix())
                for frame in reader.iter_frames(camera_id):
                    for detection in frame.detections:
                        for field in ["mask_path", "crop_path"]:
                            path = reader.asset_path(detection[field])
                            bundle.write(path, "data/" + path.relative_to(root).as_posix())
        temporary.replace(archive)
    except BaseException:
        # Keep a partial archive clearly named for diagnostics; never expose it as finished.
        raise
    return {"archive": str(archive), "size_bytes": archive.stat().st_size, "validation": report}
