#!/usr/bin/env python3
"""Replay a prepared MCAP into the Humble node on an isolated ROS domain."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from mcap.reader import make_reader
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--executable", type=Path, required=True)
    ap.add_argument("--mode", choices=["lio", "livo", "lidar_scheduled"], required=True)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--domain", type=int, default=91)
    args = ap.parse_args()
    if args.rate <= 0:
        ap.error("rate must be positive")
    folder = args.experiment.resolve()
    out = folder / args.mode
    if out.exists():
        ap.error(
            f"{out} already exists; preserve results and select another experiment"
        )
    out.mkdir()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    os.environ["ROS_LOG_DIR"] = str(out / "ros_logs")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["RCUTILS_COLORIZED_OUTPUT"] = "0"
    env = os.environ.copy()
    command = [
        str(args.executable.resolve()),
        "--ros-args",
        "--params-file",
        str(folder / (args.mode + ".yaml")),
    ]
    log = (out / "node.log").open("w")
    proc = None
    node = None
    observed = []
    sent = Counter()
    rss_peak = 0
    started = time.monotonic()
    try:
        rclpy.init()
        node = rclpy.create_node("vbot_experiment_replay")

        def observe(m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            observed.append(
                [
                    m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                    p.x,
                    p.y,
                    p.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                ]
            )

        _subscription = node.create_subscription(Odometry, "/Odometry", observe, 100)
        clock = node.create_publisher(Clock, "/clock", 10)
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        expected = {
            "/experiment/lidar": "sensor_msgs/msg/PointCloud2",
            "/experiment/imu": "sensor_msgs/msg/Imu",
        }
        if args.mode != "lio":
            expected["/experiment/image"] = "sensor_msgs/msg/Image"
        classes = {k: get_message(v) for k, v in expected.items()}
        publishers = {k: node.create_publisher(classes[k], k, qos) for k in expected}
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + 20
        while any(p.get_subscription_count() == 0 for p in publishers.values()):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"node exited at startup: {proc.returncode}; see {out}/node.log"
                )
            if time.monotonic() > deadline:
                raise RuntimeError("DDS discovery timed out")
            rclpy.spin_once(node, timeout_sec=0.05)
        print("Subscribers discovered; replay starting", flush=True)
        begin = None
        wall_begin = time.monotonic()
        last_report = wall_begin
        files = list((folder / "bag").glob("*.mcap"))
        if len(files) != 1:
            raise RuntimeError("expected one prepared MCAP")
        with files[0].open("rb") as f:
            for _, ch, record in make_reader(f).iter_messages(topics=list(expected)):
                if proc.poll() is not None:
                    raise RuntimeError(f"node exited during replay: {proc.returncode}")
                if begin is None:
                    begin = record.log_time
                target = wall_begin + (record.log_time - begin) * 1e-9 / args.rate
                while time.monotonic() < target:
                    rclpy.spin_once(
                        node, timeout_sec=min(0.01, max(0, target - time.monotonic()))
                    )
                stamp = Clock()
                stamp.clock.sec = record.log_time // 1_000_000_000
                stamp.clock.nanosec = record.log_time % 1_000_000_000
                clock.publish(stamp)
                publishers[ch.topic].publish(
                    deserialize_message(record.data, classes[ch.topic])
                )
                sent[ch.topic] += 1
                rclpy.spin_once(node, timeout_sec=0)
                if time.monotonic() - last_report > 5:
                    try:
                        status = Path(f"/proc/{proc.pid}/status").read_text()
                        rss_peak = max(
                            rss_peak,
                            int(
                                next(
                                    x.split()[1]
                                    for x in status.splitlines()
                                    if x.startswith("VmRSS:")
                                )
                            ),
                        )
                    except (FileNotFoundError, StopIteration):
                        pass
                    print(
                        f"t={(record.log_time - begin) * 1e-9:.1f}s sent={dict(sent)} odometry={len(observed)}",
                        flush=True,
                    )
                    last_report = time.monotonic()
        # Allow the last complete sensor interval to drain before stopping.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and proc.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        log.close()
    array = np.array(observed).reshape(-1, 8)
    np.savetxt(out / "observed_odometry.txt", array, fmt="%.12f")
    trajectory = np.loadtxt(out / "trajectory.txt", ndmin=2)
    if len(trajectory) < 2 or not np.all(np.isfinite(trajectory)):
        raise RuntimeError("no valid finite trajectory")
    # Keep final VIO states in LIVO, LIO states in the baseline. Same timestamps
    # in the detailed trace describe sequential filter updates, not two times.
    final = trajectory[trajectory[:, -1] == (2 if args.mode == "lio" else 1), :8]
    if len(final) < 2:
        raise RuntimeError("expected filter update type was not observed")
    np.savetxt(out / "trajectory_final.tum", final, fmt="%.12f")
    timing = np.genfromtxt(out / "timing.csv", delimiter=",", names=True)
    timings = {}
    for value, label in [(1, "visual"), (2, "lidar")]:
        a = np.atleast_1d(timing["processing_ms"])[
            np.atleast_1d(timing["update_type"]) == value
        ]
        if len(a):
            timings[label] = {
                "count": len(a),
                "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)),
            }
    metrics = {
        "mode": args.mode,
        "command": command,
        "replay_rate": args.rate,
        "domain": args.domain,
        "exit_code": proc.returncode,
        "wall_s": time.monotonic() - started,
        "sent": dict(sent),
        "observed_odometry_count": len(observed),
        "final_poses": len(final),
        "trajectory_start_s": float(final[0, 0]),
        "trajectory_end_s": float(final[-1, 0]),
        "path_length_m": float(
            np.linalg.norm(np.diff(final[:, 1:4], axis=0), axis=1).sum()
        ),
        "endpoint_displacement_m": float(
            np.linalg.norm(final[-1, 1:4] - final[0, 1:4])
        ),
        "z_range_m": float(np.ptp(final[:, 3])),
        "rss_peak_sampled_mib": rss_peak / 1024,
        "timing": timings,
        "accuracy_note": "No independent ground truth: these are execution/trajectory statistics, not ATE/RPE.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if proc.returncode != 0:
        raise RuntimeError(f"node failed: {proc.returncode}")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
