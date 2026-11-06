"""Download known public weights without executing any downloaded Python code."""

from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "yolo": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
        "path": "models/yolo11s-seg.pt",
        "sha256": "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3",
        "model": "YOLO11s-seg", "release": "v8.3.0", "format": "pytorch",
    },
    "schp": {
        "url": "https://huggingface.co/pirocheto/schp-atr-18/resolve/a54fa65e7e4f27f21011f652ffc0053ebb24b292/onnx/schp-atr-18-int8-static.onnx",
        "path": "models/schp/schp-atr-18.onnx",
        "sha256": "4420d8db8c1f266967c89485786b01209f6d405f320fc0f87e8ced49392cefb5",
        "model": "SCHP ATR18", "format": "onnx", "quantization": "INT8 static",
        "revision": "a54fa65e7e4f27f21011f652ffc0053ebb24b292",
        "processor": "PIL RGB resize 512 bilinear; mean [.406,.456,.485], std [.225,.224,.229]",
    },
}


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(name):
    import requests
    source = SOURCES[name]
    path = ROOT / source["path"]
    expected = source.get("sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = digest_file(path)
        if expected and actual != expected:
            raise ValueError(f"Existing {path} does not match the expected SHA256; inspect or move it before retrying")
        if path.stat().st_size < 1024 * 1024:
            raise ValueError(f"Existing {path} is too small to be a valid model")
        print(f"Already downloaded: {path.name}", flush=True)
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {name}: {source['url']}", flush=True)
        result = hashlib.sha256()
        with requests.get(source["url"], stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    result.update(chunk)
        actual = result.hexdigest()
        if expected and actual != expected:
            raise ValueError(f"Downloaded {name} checksum mismatch; .part retained for inspection")
        if temporary.stat().st_size < 1024 * 1024:
            raise ValueError(f"Downloaded {name} is too small to be a valid model")
        temporary.replace(path)
        print(f"Saved {path.name}: {path.stat().st_size:,} bytes, SHA256 {actual}", flush=True)
    record = {**source, "sha256": actual, "size_bytes": path.stat().st_size}
    path.with_suffix(path.suffix + ".source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(SOURCES))
    args = parser.parse_args()
    for name in ([args.only] if args.only else SOURCES):
        download(name)


if __name__ == "__main__":
    main()
