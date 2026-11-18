#!/usr/bin/env python3
"""Inventory all MCAP files and audit recorded calibration without editing bags."""

import argparse
from collections import Counter
import json
from pathlib import Path

from mcap.reader import make_reader
from prepare_vbot_bag import inspect_calibration


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    files = sorted(args.data.resolve().rglob("*.mcap"))
    if not files:
        ap.error("no MCAP recordings found")

    def bag_name(path):
        return (
            path.parent.parent.name + "_" + path.parent.name
            if path.parent.name.isdigit()
            else path.parent.name
        )

    names = Counter(bag_name(p) for p in files)
    entries = []
    for path in files:
        with path.open("rb") as stream:
            summary = make_reader(stream).get_summary()
        if summary is None or summary.statistics is None:
            raise RuntimeError(f"MCAP summary missing: {path}")
        stats = summary.statistics
        topics = {}
        for channel_id, channel in summary.channels.items():
            row = topics.setdefault(
                channel.topic,
                {"type": summary.schemas[channel.schema_id].name, "count": 0},
            )
            row["count"] += stats.channel_message_counts.get(channel_id, 0)
        entry = {
            "path": str(path),
            "name": bag_name(path) if names[bag_name(path)] == 1 else path.stem,
            "duration_s": (stats.message_end_time - stats.message_start_time) * 1e-9,
            "first_record_ns": stats.message_start_time,
            "last_record_ns": stats.message_end_time,
            "size_bytes": path.stat().st_size,
            "topics": topics,
        }
        try:
            info, transforms = inspect_calibration(
                path,
                "/stereo_left/camera_info",
                "base_link",
                "lidar",
                "lidar_imu",
                "stereo_left",
            )
            entry["calibration"] = {
                "width": info.width,
                "height": info.height,
                "K": list(info.k),
                "D": list(info.d),
                "distortion_model": info.distortion_model,
                "transforms": {
                    key: value.tolist() for key, value in transforms.items()
                },
            }
            entry["calibration_status"] = "ok"
        except Exception as error:
            entry.update(calibration_status="error", calibration_error=str(error))
        entries.append(entry)
        print(
            entry["name"], entry["duration_s"], entry["calibration_status"], flush=True
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
