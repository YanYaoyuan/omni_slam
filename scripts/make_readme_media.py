#!/usr/bin/env python3
"""Export README previews from existing experiment artifacts (requires FFmpeg).

No model inference, re-rendering of predictions, or changes to frozen results.
Original videos/reports are local-only; the small derived assets are tracked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    dict(name="garage_mapping.gif", source="docs/reports/semantic_mapping_video_2026-11-06/livo_semantic_experiment_1080p.mp4",
         kind="gif", speed=6, source_replay_speed=5, width=720, fps=5),
    dict(name="outdoor_objects_ground.gif", source="docs/reports/boxes_ground_2026-11-06/boxes_ground_1080p.mp4",
         kind="gif", speed=8, source_replay_speed=10, width=720, fps=5),
    dict(name="garage_vehicle_objects.png", source="docs/paper/visibility_revision/latex/figures/current/robot_vehicles.png",
         kind="image", filters="crop=1050:435:1050:465",
         description="Lower-right Ours candidate panel, unchanged predicted vehicle instances; no robot GT."),
    dict(name="global_semantic_ground.png", source="output/presentations/omni_slam_20261118_v2/maps/global_semantic_ground.png",
         kind="image", filters="scale=1600:-2:flags=lanczos",
         description="Xinjinqiao offline global semantic and ground map; existing presentation figure."),
    dict(name="object_query_3d.png", source="docs/reports/boxes_ground_2026-11-06/viewer_door.png",
         kind="image", filters="scale=1280:-2:flags=lanczos",
         description="Existing viewer screenshot: door observations, 3D boxes, coordinates and image evidence."),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "format=duration,size:stream=width,height,nb_read_frames",
        "-of", "json", str(path),
    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/readme")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for spec in ASSETS:
        if not (ROOT / spec["source"]).is_file():
            parser.error(f"Local source missing: {spec['source']}")
        if (args.output / spec["name"]).exists() and not args.overwrite:
            parser.error("Output exists; use --overwrite to regenerate previews")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in ASSETS:
        source = ROOT / spec["source"]
        target = args.output / spec["name"]
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1",
                   "-i", str(source)]
        if spec["kind"] == "gif":
            filters = (
                f"[0:v]setpts=(PTS-STARTPTS)/{spec['speed']},fps={spec['fps']},"
                f"scale={spec['width']}:-2:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=96:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=none:diff_mode=rectangle[v]"
            )
            command += ["-filter_complex", filters, "-map", "[v]", "-an", "-loop", "0"]
        else:
            command += ["-vf", spec["filters"], "-frames:v", "1", "-update", "1"]
        command.append(str(target))
        subprocess.run(command, check=True)
        info = probe(target)
        if spec["kind"] == "gif":
            assert int(info["streams"][0]["nb_read_frames"]) > 1
            assert target.stat().st_size < 12 * 1024 * 1024, "GIF exceeds README budget"
        records.append(dict(spec, source_sha256=sha256(source),
                            output_sha256=sha256(target), output_bytes=target.stat().st_size,
                            output_probe=info))
        print(f"{target.name}: {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    manifest = dict(
        description="Existing experiment outputs; previews only. GIF speed is relative to already accelerated source video, not processing throughput.",
        ffmpeg_version=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        assets=records,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
