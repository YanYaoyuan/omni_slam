#!/usr/bin/env python3
"""Real ROS replay: Manager -> LIVO -> TF/body odometry -> SaveMap, with image outage."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from mcap.reader import make_reader
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger
from omni_slam_interfaces.msg import LivoKeyframe
from omni_slam_interfaces.srv import StartMapping, StopMapping
from omni_slam_interfaces.action import SaveMap
from omni_tf_manager.msg import SlamStatus


def ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def matrix(pose):
    q = pose.orientation
    p = pose.position
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    t[:3, 3] = [p.x, p.y, p.z]
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ["bag", "config", "tf-config", "output"]:
        ap.add_argument("--" + arg, type=Path, required=True)
    args = ap.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        ap.error("output exists")
    args.output.mkdir(parents=True)
    procs = []
    logs = []

    def start(command, name):
        log = (args.output / (name + ".log")).open("w")
        logs.append(log)
        p = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        procs.append(p)
        return p

    rclpy.init()
    node = rclpy.create_node("livo_integration_probe")
    sensor = {}
    body = {}
    clouds = []
    keys = []
    states = []
    counts = Counter()
    node.create_subscription(
        Odometry,
        "/state_estimation",
        lambda m: sensor.__setitem__(ns(m.header.stamp), matrix(m.pose.pose)),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        Odometry,
        "/omni/tf_manager/body_odom",
        lambda m: body.__setitem__(ns(m.header.stamp), matrix(m.pose.pose)),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        PointCloud2,
        "/cloud_registered_global",
        lambda m: clouds.append(
            (ns(m.header.stamp), m.width * m.height, m.header.frame_id)
        ),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        LivoKeyframe,
        "/omni/slam/livo/keyframes",
        lambda m: keys.append(
            (
                ns(m.header.stamp),
                m.id,
                m.visual_update,
                ns(m.image_stamp),
                m.cloud.width * m.cloud.height,
                m.header.frame_id,
                m.cloud.header.frame_id,
            )
        ),
        10,
    )
    node.create_subscription(
        SlamStatus,
        "/omni/slam/status",
        lambda m: states.append((m.state, m.reason_text)),
        10,
    )
    clock = node.create_publisher(Clock, "/clock", 10)
    types = {
        "/experiment/image": Image,
        "/experiment/imu": Imu,
        "/experiment/lidar": PointCloud2,
    }
    names = {
        "/experiment/image": "/omni/sensors/camera/image_rect",
        "/experiment/imu": "/omni/sensors/imu/data",
        "/experiment/lidar": "/omni/sensors/lidar/points",
    }
    publishers = {
        k: node.create_publisher(types[k], names[k], qos_profile_sensor_data)
        for k in types
    }

    def spin(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            rclpy.spin_once(
                node, timeout_sec=min(0.02, max(0, until - time.monotonic()))
            )

    def result(future, seconds=30):
        end = time.monotonic() + seconds
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not future.done():
            raise RuntimeError("ROS operation timed out")
        return future.result()

    try:
        start(
            [
                "ros2",
                "run",
                "omni_tf_manager",
                "omni_tf_manager_node",
                "--ros-args",
                "--params-file",
                str(args.tf_config.resolve()),
            ],
            "tf",
        )
        start(
            [
                "python3",
                "-c",
                "from omni_slam_manager.manager_node import main; main()",
                "--ros-args",
                "-p",
                "robot_profile:=vbot",
                "-p",
                "mapping_backend:=fast_livo",
                "-p",
                "use_sim_time:=true",
                "-p",
                "mapping_config_file:=" + str(args.config.resolve()),
                "-p",
                "tf_profile_path:=" + str(args.tf_config.resolve()),
                "-p",
                "map_root:=" + str(args.output / "maps"),
                "-p",
                "map_source_path:=" + str(args.output / "work/map.pcd"),
            ],
            "manager",
        )
        client = node.create_client(StartMapping, "/omni/slam/start_mapping")
        if not client.wait_for_service(timeout_sec=15):
            raise RuntimeError("manager unavailable")
        request = StartMapping.Request()
        request.session_id = "livo_integration"
        response = result(client.call_async(request))
        assert response.accepted, response.message
        save = node.create_client(Trigger, "/map_save")
        if not save.wait_for_service(timeout_sec=20):
            raise RuntimeError("LIVO save endpoint unavailable")
        empty = result(save.call_async(Trigger.Request()))
        assert not empty.success
        spin(1)
        first = None
        wall = None
        with args.bag.open("rb") as stream:
            for _, channel, record in make_reader(stream).iter_messages():
                if channel.topic not in types:
                    continue
                stamp = record.log_time
                if first is None:
                    first = stamp
                    wall = time.monotonic()
                elapsed = (stamp - first) * 1e-9
                delay = wall + elapsed - time.monotonic()
                if delay > 0:
                    spin(delay)
                tick = Clock()
                tick.clock.sec = stamp // 10**9
                tick.clock.nanosec = stamp % 10**9
                clock.publish(tick)
                if channel.topic == "/experiment/image" and 12 <= elapsed < 22:
                    counts["withheld_images"] += 1
                    continue
                msg = deserialize_message(record.data, types[channel.topic])
                publishers[channel.topic].publish(msg)
                counts[channel.topic] += 1
                rclpy.spin_once(node, timeout_sec=0)
        spin(0.5)
        assert any(s == 2 for s, _ in states), states[-5:]
        action = ActionClient(node, SaveMap, "/omni/slam/save_map")
        assert action.wait_for_server(timeout_sec=5)
        goal = SaveMap.Goal()
        goal.map_id = "livo_test"
        goal.make_current = True
        goal.calibration_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
        handle = result(action.send_goal_async(goal))
        assert handle.accepted
        saved = result(handle.get_result_async()).result
        assert saved.success, saved.message
        stop = node.create_client(StopMapping, "/omni/slam/stop_mapping")
        assert stop.wait_for_service(timeout_sec=5)
        stopped = result(stop.call_async(StopMapping.Request()))
        assert stopped.accepted, stopped.message
        spin(0.5)
        common = sorted(sensor.keys() & body.keys())
        assert len(common) > 20
        params = yaml.safe_load(args.tf_config.read_text())["omni_tf_manager"][
            "ros__parameters"
        ]
        bi = np.eye(4)
        bi[:3, :3] = Rotation.from_euler(
            "xyz", params["static_transform.imu_mount.rotation_rpy"]
        ).as_matrix()
        bi[:3, 3] = params["static_transform.imu_mount.translation"]
        error = max(np.max(np.abs(body[t] @ bi - sensor[t])) for t in common)
        assert error < 1e-6, error

        def gap(t):
            return 13 < (t - first) * 1e-9 < 21

        gap_poses = sum(gap(t) for t in sensor)
        gap_clouds = [x for x in clouds if gap(x[0])]
        assert (
            gap_poses > 20
            and len(gap_clouds) > 20
            and all(c[1] > 0 for c in gap_clouds)
        )
        assert any(not k[2] and gap(k[0]) for k in keys)
        assert any(k[2] and (k[0] - first) * 1e-9 > 23 for k in keys)
        assert all(
            k[4] > 0 and k[5] == "omni_odom" and k[6] == "omni_imu_link" for k in keys
        )
        assert all(0 <= k[0] - k[3] <= 40_000_001 for k in keys if k[2])
        assert len({k[1] for k in keys}) == len(keys)
        report = dict(
            status="pass",
            input_counts=dict(counts),
            sensor_poses=len(sensor),
            body_poses=len(body),
            matched_body_poses=len(common),
            body_transform_max_error=error,
            outage_poses=gap_poses,
            outage_nonempty_clouds=len(gap_clouds),
            keyframes=len(keys),
            visual_keyframes=sum(k[2] for k in keys),
            mapping_state_seen=True,
            map_saved=saved.map_path,
            map_checksum=saved.map_checksum,
            stopped=stopped.accepted,
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        (args.output / "keyframes.json").write_text(json.dumps(keys))
        print(json.dumps(report, indent=2))
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
