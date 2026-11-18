#!/usr/bin/env python3
"""Source-only trial bundle; no bags, reports, learned weights or host binaries."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output exists")
    root = Path(__file__).resolve().parents[1]
    roots = ["omni_slam_interfaces", "omni_slam_manager", "omni_semantic_map",
             "FAST_LIVO2/CMakeLists.txt", "FAST_LIVO2/package.xml", "FAST_LIVO2/LICENSE",
             "FAST_LIVO2/UPSTREAM.md", "FAST_LIVO2/include", "FAST_LIVO2/src",
             "FAST_LIVO2/README.md", "FAST_LIVO2/README_UPSTREAM.md",
             "FAST_LIVO2/config", "FAST_LIVO2/launch", "FAST_LIVO2/third_party",
             "FAST_LIVO2/scripts", "FAST_LIVO2/test", "scripts/build_livo_trial.sh",
             "scripts/record_livo_trial.sh", "scripts/package_livo_trial.py",
             "docs/experiments/s100_trial_preparation_2026-11-06.md",
             "docs/deployment/s100_livo_semantic_trial.md"]
    files = []
    excluded = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv",
                "build", "install", "log", "vikit_ros", "vikit_py"}
    for name in roots:
        path = root / name
        for source in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if source.is_symlink() or not source.is_file() or excluded.intersection(source.parts):
                continue
            if any(part.endswith(".egg-info") for part in source.parts):
                continue
            if source.suffix in (".pyc", ".pcd", ".mcap", ".db3", ".mp4", ".pth",
                                 ".pt", ".ckpt", ".safetensors", ".onnx", ".engine", ".so", ".a"):
                continue
            files.append(source)
    if not (root / roots[-1]).is_file():
        raise FileNotFoundError(roots[-1])
    manifest = {"kind": "source_only", "created_unix": time.time(),
        "board": "S100 native Humble LIVO", "semantic_compute": "companion workstation",
        "hardware_verified": False, "files": {str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for p in files:
            archive.add(p, arcname="omni_slam_trial/"+str(p.relative_to(root)), recursive=False)
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("omni_slam_trial/manifest.json"); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(f"{args.output}: {len(files)} files, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
