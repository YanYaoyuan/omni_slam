#!/usr/bin/env python3
"""Create a bounded experiment MCAP from VBot H265/LiDAR/IMU recordings.

Calibration comes from CameraInfo and recorded TF, never another robot's YAML.
H265 PTS are copied from the CompressedVideo timestamp; rational distortion is
removed before the pinhole VIO model. Input recordings are opened read-only.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def time_ns(t):
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


def matrix(tf):
    q = tf.rotation
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    v = tf.translation
    out[:3, 3] = [v.x, v.y, v.z]
    return out


def inspect_calibration(path, image_info, base, lidar_frame, imu_frame, camera_frame):
    info = None
    transforms = {}
    with path.open("rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, msg in r.iter_decoded_messages(topics=[image_info, "/tf_static"]):
            if ch.topic == image_info:
                if info is None:
                    info = msg
                elif list(info.k) != list(msg.k) or list(info.d) != list(msg.d):
                    raise ValueError(
                        "CameraInfo changed during recording; split by calibration version"
                    )
            else:
                for tf in msg.transforms:
                    if tf.header.frame_id != base:
                        continue
                    t = matrix(tf.transform)
                    if tf.child_frame_id in transforms and not np.allclose(
                        transforms[tf.child_frame_id], t, atol=1e-7
                    ):
                        raise ValueError("static calibration changed during recording")
                    transforms[tf.child_frame_id] = t
    if info is None:
        raise ValueError(f"missing {image_info}")
    for frame in (lidar_frame, imu_frame, camera_frame):
        if frame not in transforms:
            raise ValueError(f"missing direct static transform {base} -> {frame}")
    if info.header.frame_id != camera_frame:
        raise ValueError(
            "CameraInfo frame does not match the selected calibration frame"
        )
    return info, transforms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="input .mcap file or bag directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--skip", type=float, default=0.0)
    ap.add_argument(
        "--image-stride",
        type=int,
        default=3,
        help="decode every frame, retain every Nth",
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--image-topic", default="/image_left_raw/h265")
    ap.add_argument("--camera-info-topic", default="/stereo_left/camera_info")
    ap.add_argument("--lidar-topic", default="/lidar_points")
    ap.add_argument("--imu-topic", default="/lidar_imu")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--lidar-frame", default="lidar")
    ap.add_argument("--imu-frame", default="lidar_imu")
    ap.add_argument("--camera-frame", default="stereo_left")
    args = ap.parse_args()
    if (
        min(args.duration, args.image_stride, args.width, args.height) <= 0
        or args.skip < 0
    ):
        ap.error(
            "duration, dimensions and stride must be positive; skip must be nonnegative"
        )
    path = args.input.resolve()
    if path.is_dir():
        files = sorted(path.glob("*.mcap"))
        if len(files) != 1:
            ap.error(
                "provide a single MCAP file; multi-file concatenation is not supported"
            )
        path = files[0]
    output = args.output.resolve()
    if output.exists():
        ap.error("output already exists; choose a new experiment directory")
    info, tf = inspect_calibration(
        path,
        args.camera_info_topic,
        args.base_frame,
        args.lidar_frame,
        args.imu_frame,
        args.camera_frame,
    )
    K = np.array(info.k).reshape(3, 3)
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError("uncalibrated camera")
    if info.distortion_model not in ["rational_polynomial", "plumb_bob"]:
        raise ValueError("unsupported camera distortion model")
    if abs(args.width / args.height - info.width / info.height) > 1e-6:
        raise ValueError("output aspect ratio must match CameraInfo")
    Kout = K.copy()
    Kout[0, :] *= args.width / info.width
    Kout[1, :] *= args.height / info.height
    map1, map2 = cv2.initUndistortRectifyMap(
        K, np.array(info.d), np.eye(3), Kout, (args.width, args.height), cv2.CV_32FC1
    )
    T_i_l = np.linalg.inv(tf[args.imu_frame]) @ tf[args.lidar_frame]
    T_c_l = np.linalg.inv(tf[args.camera_frame]) @ tf[args.lidar_frame]
    output.mkdir(parents=True)
    # Imported after setup.bash so that ROS native libraries can be located.
    import rosbag2_py
    from rclpy.serialization import serialize_message, deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import Image

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output / "bag"), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {
        "/experiment/lidar": "sensor_msgs/msg/PointCloud2",
        "/experiment/imu": "sensor_msgs/msg/Imu",
        "/experiment/image": "sensor_msgs/msg/Image",
    }
    for name, type_name in types.items():
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=name, type=type_name, serialization_format="cdr"
            )
        )
    codec = av.CodecContext.create("hevc", "r")
    codec.thread_count = 1
    counts = Counter()
    times = defaultdict(list)
    offsets = defaultdict(list)
    errors = []
    sample = None
    selected = [
        args.lidar_topic,
        args.imu_topic,
        args.image_topic,
        "/odometry",
        "/gnss/fix",
    ]
    reference = []
    gnss_status = Counter()
    decode_count = 0
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        start = summary.statistics.message_start_time + int(args.skip * 1e9)
        stop = start + int(args.duration * 1e9)
        # Decode from the beginning to obtain codec parameter sets/keyframes.
        for schema, ch, record, decoded in reader.iter_decoded_messages(
            topics=selected
        ):
            if record.log_time > stop + 2_000_000_000:
                break
            if ch.topic == args.image_topic:
                if decoded.format not in ("h265", "hevc"):
                    raise ValueError("selected video is not H265")
                stamp = time_ns(decoded.timestamp)
                if stamp > stop:
                    continue
                pkt = av.Packet(bytes(decoded.data))
                pkt.pts = stamp
                pkt.dts = stamp
                pkt.time_base = Fraction(1, 1_000_000_000)
                try:
                    frames = codec.decode(pkt)
                except av.error.FFmpegError as e:
                    counts["video_decode_errors"] += 1
                    if len(errors) < 5:
                        errors.append(str(e))
                    continue
                for frame in frames:
                    if frame.pts is None:
                        raise ValueError(
                            "decoder returned an image without a source timestamp"
                        )
                    pts = int(Fraction(frame.pts) * frame.time_base * 1_000_000_000)
                    decode_count += 1
                    if (
                        pts < start
                        or pts > stop
                        or (decode_count - 1) % args.image_stride
                    ):
                        continue
                    if (frame.width, frame.height) != (info.width, info.height):
                        raise ValueError(
                            "decoded image dimensions differ from CameraInfo"
                        )
                    image = cv2.remap(
                        frame.to_ndarray(format="bgr24"), map1, map2, cv2.INTER_LINEAR
                    )
                    msg = Image()
                    msg.header.stamp.sec = pts // 1_000_000_000
                    msg.header.stamp.nanosec = pts % 1_000_000_000
                    msg.header.frame_id = "omni_rgb_camera_optical_frame"
                    msg.width = args.width
                    msg.height = args.height
                    msg.encoding = "bgr8"
                    msg.step = args.width * 3
                    msg.data = image.tobytes()
                    writer.write("/experiment/image", serialize_message(msg), pts)
                    counts["images"] += 1
                    times["image"].append(pts)
                    offsets["image"].append((record.log_time - stamp) * 1e-6)
                    if sample is None:
                        sample = image
                        cv2.imwrite(str(output / "sample_undistorted.jpg"), image)
            else:
                stamp = time_ns(decoded.header.stamp)
                if not (start <= stamp <= stop):
                    continue
                if ch.topic == "/odometry":
                    p = decoded.pose.pose.position
                    q = decoded.pose.pose.orientation
                    reference.append([stamp * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                    continue
                if ch.topic == "/gnss/fix":
                    gnss_status[int(decoded.status.status)] += 1
                    continue
                kind = "lidar" if ch.topic == args.lidar_topic else "imu"
                # Only rewrite the known VBot frame aliases; do not transform payloads.
                msg = deserialize_message(record.data, get_message(schema.name))
                msg.header.frame_id = (
                    "omni_lidar_link" if kind == "lidar" else "omni_imu_link"
                )
                writer.write("/experiment/" + kind, serialize_message(msg), stamp)
                counts[kind] += 1
                times[kind].append(stamp)
                offsets[kind].append((record.log_time - stamp) * 1e-6)
    del writer
    if not all(counts[k] > 0 for k in ["images", "lidar", "imu"]):
        raise ValueError("missing required decoded experiment stream")
    np.savetxt(
        output / "vendor_odometry.txt", np.array(reference).reshape(-1, 8), fmt="%.12f"
    )
    params = {
        "use_sim_time": True,
        "common": {
            "lid_topic": "/experiment/lidar",
            "imu_topic": "/experiment/imu",
            "img_topic": "/experiment/image",
            "img_en": 1,
            "lidar_en": 1,
            "odom_frame": "omni_odom",
            "tracking_frame": "omni_imu_link",
            "camera_frame": "omni_rgb_camera_optical_frame",
        },
        "camera": {
            "width": args.width,
            "height": args.height,
            "scale": 1.0,
            "fx": float(Kout[0, 0]),
            "fy": float(Kout[1, 1]),
            "cx": float(Kout[0, 2]),
            "cy": float(Kout[1, 2]),
            "distortion": [0.0] * 5,
        },
        "extrin_calib": {
            "extrinsic_T": T_i_l[:3, 3].tolist(),
            "extrinsic_R": T_i_l[:3, :3].ravel().tolist(),
            "Rcl": T_c_l[:3, :3].ravel().tolist(),
            "Pcl": T_c_l[:3, 3].tolist(),
        },
        "time_offset": {
            "img_time_offset": 0.0,
            "imu_time_offset": 0.0,
            "lidar_time_offset": 0.0,
        },
        "preprocess": {
            "lidar_type": 8,
            "scan_line": 16,
            "point_filter_num": 1,
            "blind": 0.3,
            "filter_size_surf": 0.2,
        },
        "imu": {"imu_en": True, "imu_int_frame": 20, "acc_cov": 0.1, "gyr_cov": 0.1},
        "vio": {
            "grid_size": 20,
            "grid_n_height": 18,
            "patch_size": 8,
            "patch_pyrimid_level": 3,
            "max_iterations": 5,
            "img_point_cov": 100.0,
            "outlier_threshold": 1000.0,
            "normal_en": True,
            "raycast_en": False,
            "exposure_estimate_en": True,
            "inv_expo_cov": 0.1,
        },
        "lio": {
            "max_iterations": 5,
            "voxel_size": 0.5,
            "max_layer": 2,
            "layer_init_num": [5, 5, 5],
            "max_points_num": 50,
            "min_eigen_value": 0.0025,
        },
        "publish": {
            "tf_en": False,
            "pub_scan_num": 1,
            "dense_map_en": False,
            "pub_plane_en": False,
        },
        "pcd_save": {"pcd_save_en": True, "interval": -1, "filter_size_pcd": 0.15},
        "runtime": {"output_directory": str(output / "livo")},
    }
    for mode, img_en in [("livo", 1), ("lio", 0), ("lidar_scheduled", 1)]:
        params["common"]["img_en"] = img_en
        params["runtime"]["output_directory"] = str(output / mode)
        params["vio"]["update_enabled"] = mode != "lidar_scheduled"
        (output / (mode + ".yaml")).write_text(
            yaml.safe_dump({"/**": {"ros__parameters": params}}, sort_keys=False)
        )

    def statistics(v):
        a = np.asarray(v, dtype=np.int64)
        delta = np.diff(a) * 1e-9
        return {
            "count": len(a),
            "first_ns": int(a[0]),
            "last_ns": int(a[-1]),
            "rate_hz": float(1 / np.median(delta)) if len(delta) else None,
            "max_gap_ms": float(np.max(delta) * 1e3) if len(delta) else None,
            "nonmonotonic": int(np.sum(delta <= 0)),
        }

    report = {
        "source": str(path),
        "source_size_bytes": path.stat().st_size,
        "duration_requested_s": args.duration,
        "counts": dict(counts),
        "decode_errors": errors,
        "streams": {k: statistics(v) for k, v in times.items()},
        "record_minus_sensor_ms": {
            k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
            for k, v in offsets.items()
        },
        "camera": {
            "source_frame": info.header.frame_id,
            "width": info.width,
            "height": info.height,
            "K": list(info.k),
            "D": list(info.d),
            "distortion_model": info.distortion_model,
            "output_K": Kout.tolist(),
        },
        "static_tf": {k: v.tolist() for k, v in tf.items()},
        "T_imu_lidar": T_i_l.tolist(),
        "T_camera_lidar": T_c_l.tolist(),
        "gnss_status_counts": dict(gnss_status),
        "reference": "vendor odometry, not independent ground truth",
        "timestamp_policy": "acquisition timestamps; zero offsets are an unverified initial hypothesis",
    }
    (output / "input_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": dict(counts),
                "gnss_status": dict(gnss_status),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
