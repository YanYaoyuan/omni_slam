#!/usr/bin/env python3
"""Replay only LiDAR/IMU while the image-enabled full-scan fusion stays active."""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess

from mcap.reader import make_reader
import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, required=True, help="completed experiment bag folder"
    )
    ap.add_argument("--executable", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output exists")
    args.output.mkdir(parents=True)
    params = yaml.safe_load((args.source / "livo.yaml").read_text())
    cfg = params["/**"]["ros__parameters"]
    assert cfg["common"]["img_en"] == 1 and cfg["vio"]["lidar_timed_fusion"]
    cfg["runtime"]["output_directory"] = str(args.output.resolve())
    config = args.output / "livo.yaml"
    config.write_text(yaml.safe_dump(params))
    audit = json.loads((args.source / "input_audit.json").read_text())["source"]
    start = audit["first_record_ns"]
    events = []
    counts = [0, 0, 0]
    with Path(audit["path"]).open("rb") as stream:
        for _, ch, m in make_reader(stream).iter_messages(
            topics=["/lidar_imu", "/lidar_points"]
        ):
            if m.log_time > start + 10_000_000_000:
                break
            sec, nsec = struct.unpack_from("<iI", m.data, 4)
            stamp = sec * 1_000_000_000 + nsec
            if stamp < start:
                continue
            kind = 1 if ch.topic == "/lidar_imu" else 2
            events.append((stamp, len(events), kind, m.data))
            counts[kind - 1] += 1
    env = os.environ.copy()
    env.update(
        ROS_DOMAIN_ID="171",
        ROS_LOCALHOST_ONLY="1",
        ROS_LOG_DIR=str(args.output.resolve() / "ros_logs"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="3",
    )
    with (args.output / "node.log").open("wb") as log:
        with subprocess.Popen(
            [
                str(args.executable.resolve()),
                "--ros-args",
                "--params-file",
                str(config.resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        ) as proc:
            proc.stdin.write(b"OMNILIVO1")
            for _, _, kind, payload in sorted(events):
                proc.stdin.write(struct.pack("<BI", kind, len(payload)) + payload)
            proc.stdin.write(struct.pack("<BIQQQ", 0, 24, *counts))
            proc.stdin.close()
            if proc.wait(timeout=60):
                raise RuntimeError("outage worker failed")
    transport = json.loads((args.output / "offline_transport.json").read_text())
    a = np.loadtxt(args.output / "trajectory.txt", ndmin=2)
    b = np.loadtxt(args.source / "lio/trajectory_final.tum", ndmin=2)
    _, ia, ib = np.intersect1d(a[:, 0], b[:, 0], return_indices=True)
    assert transport["completed"] and transport["input_counts"]["image"] == 0
    assert len(a) > 2 and np.all(a[:, -1] == 2) and len(ia) == len(a)
    # TUM export rounds the native 17-digit values to 12 decimal places.
    assert np.allclose(a[ia, :8], b[ib], rtol=0, atol=1e-11)
    result = {
        "camera_configured_enabled": True,
        "delivered_images": 0,
        "poses": len(a),
        "same_as_pure_lio": True,
        "source_complete": False,
        "duration_s": 10,
        "transport": transport,
    }
    (args.output / "outage_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
