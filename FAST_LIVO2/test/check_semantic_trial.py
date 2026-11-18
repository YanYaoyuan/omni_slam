#!/usr/bin/env python3
"""Native LIVO -> atomic packet -> real local inference -> ROS boxes/ground smoke test."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from omni_slam_interfaces.msg import LivoSemanticFrame
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("bag", "config", "weights", "ground-model", "python", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    args = ap.parse_args(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    procs = []; logs = []; counts = Counter(); states = []; packets = []
    report = {"passed": False}
    def start(command, name):
        log = (args.output / (name+".log")).open("w"); logs.append(log)
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(proc); return proc
    rclpy.init(); node = rclpy.create_node("semantic_trial_probe")
    def packet(msg):
        counts["packets"] += 1
        k = msg.keyframe
        age = (k.header.stamp.sec-msg.image.header.stamp.sec)+(k.header.stamp.nanosec-msg.image.header.stamp.nanosec)*1e-9
        assert k.visual_update and -.001 <= age <= .1
        assert k.image_stamp == msg.image.header.stamp
        assert k.cloud.header.frame_id == k.tracking_frame
        assert msg.camera_info.width == msg.image.width == 640
        assert np.isfinite(msg.camera_info.k).all() and np.all(np.asarray(msg.camera_info.d) == 0)
        packets.append(dict(id=int(k.id), age_ms=age*1000, points=k.cloud.width*k.cloud.height))
        if counts["packets"] <= 3:
            (args.output / f"packet_{counts['packets']}.cdr").write_bytes(serialize_message(msg))
    def boxes(msg):
        counts["box_messages"] += 1
        counts["line_boxes"] += sum(m.type == m.LINE_LIST and m.action == m.ADD for m in msg.markers)
    def ground(msg):
        counts["ground_messages"] += 1; counts["ground_points"] += msg.width*msg.height
    node.create_subscription(LivoSemanticFrame, "/omni/slam/livo/semantic_frames", packet, qos_profile_sensor_data)
    node.create_subscription(MarkerArray, "/omni/semantic/boxes", boxes, qos_profile_sensor_data)
    node.create_subscription(PointCloud2, "/omni/semantic/ground", ground, qos_profile_sensor_data)
    node.create_subscription(String, "/omni/semantic/status", lambda m: states.append(json.loads(m.data)), 10)
    def spin_for(seconds, stop=None):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
            if stop and stop():
                return
    try:
        semantic = start([str(args.python), "-m", "omni_semantic_map.live_node", "--ros-args",
            "-p", f"weights:={args.weights.resolve()}", "-p", f"ground_model:={args.ground_model.resolve()}",
            "-p", f"output_directory:={args.output/'semantic'}", "-p", "use_sim_time:=true"], "semantic")
        spin_for(40, lambda: bool(states) or semantic.poll() is not None)
        assert semantic.poll() is None and states, "semantic node did not start"
        livo = start(["ros2", "launch", "fast_livo", "managed_mapping.launch.py",
            f"config:={args.config.resolve()}", f"output_directory:={args.output/'livo'}",
            f"map_path:={args.output/'map.pcd'}", "use_sim_time:=true"], "livo")
        spin_for(3)
        player = start(["ros2", "bag", "play", str(args.bag.resolve()), "--clock", "50", "-r", ".5",
            "--disable-keyboard-controls", "--remap",
            "/experiment/lidar:=/omni/sensors/lidar/points",
            "/experiment/imu:=/omni/sensors/imu/data",
            "/experiment/image:=/omni/sensors/camera/image_rect"], "player")
        spin_for(95, lambda: player.poll() is not None or semantic.poll() is not None)
        assert player.poll() == 0, "bag playback failed or timed out"
        spin_for(12)
        assert semantic.poll() is None and livo.poll() is None
        assert counts["packets"] >= 5 and counts["line_boxes"] > 0 and counts["ground_messages"] > 0, dict(counts)
        assert any(s["state"] == "no_recent_visual_keyframes" for s in states)
        assert not any(s["state"] in ("error", "session_mismatch") for s in states), states[-1]
        report.update(passed=True, counts=dict(counts), packets=packets, last_status=states[-1],
            states=sorted({s["state"] for s in states}), hardware="local x86 ROS replay, not S100")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
        for log in logs:
            log.close()
        (args.output / "validation.json").write_text(json.dumps(report, indent=2))
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
