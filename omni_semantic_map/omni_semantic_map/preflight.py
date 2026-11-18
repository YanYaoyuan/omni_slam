"""Read-only live input audit. A recorded calibration is never silently promoted."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import yaml


def audit_config(config, tf_profile):
    from omni_slam_manager.livo_profile import validate_livo_profile
    p = validate_livo_profile(config, tf_profile)
    tf = yaml.safe_load(Path(tf_profile).read_text())["omni_tf_manager"]["ros__parameters"]
    failures = []
    if p.get("use_sim_time", False):
        failures.append("live configuration has use_sim_time=true")
    if tf.get("use_sim_time", False):
        failures.append("live TF profile has use_sim_time=true")
    for config_key, tf_key in (("odom_frame", "frames.odom"), ("tracking_frame", "frames.tracking")):
        if p["common"][config_key] != tf[tf_key]:
            failures.append(f"LIVO {config_key} disagrees with TF profile")
    if p.get("publish", {}).get("tf_en", False):
        failures.append("LIVO TF publication must remain disabled")
    if not p.get("runtime", {}).get("keyframe_archive", False):
        failures.append("semantic input requires runtime.keyframe_archive=true")
    if np.any(np.asarray(p["camera"].get("distortion", [])) != 0):
        failures.append("trial expects upstream-rectified image and zero camera distortion")
    if p["camera"].get("scale", 1.) != 1.:
        failures.append("trial expects scale=1; supply calibrated input at configured resolution")
    for name in ("lid_topic", "imu_topic", "img_topic"):
        topic = p["common"][name]
        if not topic.startswith("/") or topic.startswith("/experiment/"):
            failures.append(f"{name} is not an explicit live topic: {topic}")
    return p, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tf-profile", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--observe-seconds", type=float, default=10.)
    ap.add_argument("--camera-info-topic", default="/omni/sensors/camera/camera_info")
    args = ap.parse_args()
    if args.observe_seconds < 0:
        ap.error("observe-seconds must be nonnegative")
    report = dict(platform=platform.platform(), architecture=platform.machine(),
        ros_distro=os.environ.get("ROS_DISTRO"), ros_domain_id=os.environ.get("ROS_DOMAIN_ID", "0"),
        free_disk_gb=round(shutil.disk_usage(args.output.parent.resolve()).free/1e9, 2),
        failures=[], sensors={}, ready=False,
        note="Read-only input audit; camera mount/time calibration and projection need on-site verification.")
    try:
        params, failures = audit_config(args.config, args.tf_profile)
        report["failures"].extend(failures)
        if args.observe_seconds == 0:
            report["failures"].append("live sensor observation not performed")
        else:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import PointCloud2, Imu, Image, CameraInfo
            rclpy.init(); node = Node("omni_semantic_preflight")
            subscriptions = []; samples = {}; infos = []
            def callback(role, message):
                stamp = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
                age = node.get_clock().now().nanoseconds*1e-9-stamp
                record = dict(stamp=stamp, age_s=age, frame=message.header.frame_id)
                if role == "image":
                    record.update(width=message.width, height=message.height, encoding=message.encoding)
                if role == "lidar":
                    record["fields"] = [f.name for f in message.fields]
                samples.setdefault(role, []).append(record)
            try:
                for role, topic, cls in [("lidar", params["common"]["lid_topic"], PointCloud2),
                        ("imu", params["common"]["imu_topic"], Imu),
                        ("image", params["common"]["img_topic"], Image)]:
                    subscriptions.append(node.create_subscription(cls, topic,
                        lambda msg, role=role: callback(role, msg), qos_profile_sensor_data))
                subscriptions.append(node.create_subscription(CameraInfo, args.camera_info_topic,
                    infos.append, qos_profile_sensor_data))
                end = time.monotonic()+args.observe_seconds
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=.1)
                expected_frames = dict(lidar="omni_lidar_link", imu=params["common"]["tracking_frame"],
                                       image=params["common"]["camera_frame"])
                tf = yaml.safe_load(args.tf_profile.read_text())["omni_tf_manager"]["ros__parameters"]
                expected_frames["lidar"] = tf["frames.icp_sensor"]
                for role in ("lidar", "imu", "image"):
                    data = samples.get(role, [])
                    report["sensors"][role] = {"messages": len(data)}
                    if len(data) < 2:
                        report["failures"].append(f"{role}: fewer than two live messages")
                        continue
                    stamps = np.array([x["stamp"] for x in data]); delta = np.diff(stamps)
                    ages = np.array([x["age_s"] for x in data])
                    report["sensors"][role].update(hz=float(1/np.median(delta)) if np.all(delta>0) else 0,
                        age_p95_s=float(np.quantile(ages,.95)), last=data[-1])
                    if np.any(delta <= 0) or stamps[0] <= 0 or ages.max() > 1 or ages.min() < -.1:
                        report["failures"].append(f"{role}: stale/future/nonmonotonic clock")
                    if any(x["frame"] != expected_frames[role] for x in data):
                        report["failures"].append(f"{role}: unexpected frame_id")
                image = samples.get("image", [])
                c = params["camera"]
                if image and any((x["width"],x["height"]) != (c["width"],c["height"]) or x["encoding"] not in ("bgr8","rgb8") for x in image):
                    report["failures"].append("image size/encoding differs from calibrated input")
                lidar = samples.get("lidar", [])
                if lidar and not {"x","y","z","intensity","ring","timestamp"}.issubset(lidar[-1]["fields"]):
                    report["failures"].append("Vanjee input missing expected point fields")
                if not infos:
                    report["failures"].append("no rectified CameraInfo observed")
                else:
                    info = infos[-1]
                    expected = [c["fx"],0,c["cx"],0,c["fy"],c["cy"],0,0,1]
                    if ((info.width,info.height) != (c["width"],c["height"])
                            or not np.allclose(info.k, expected, atol=1e-5)
                            or np.any(np.asarray(info.d) != 0)
                            or info.header.frame_id != expected_frames["image"]):
                        report["failures"].append("live CameraInfo differs from LIVO calibration")
                report["ready"] = not report["failures"]
            finally:
                node.destroy_node(); rclpy.shutdown()
    except Exception as exc:
        report["failures"].append(str(exc))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 2)
