"""Check that LIVO calibration and the sole TF authority describe one rig."""

from pathlib import Path
import math
import numpy as np
import yaml


LIVO_MAPPING_COMMAND = (
    "ros2 launch fast_livo managed_mapping.launch.py "
    "config:={mapping_config_file} map_path:={map_source_path} "
    "output_directory:={runtime_directory} odom_frame:={odom_frame} "
    "tracking_frame:={tracking_frame} use_sim_time:={use_sim_time}"
)


def validate_livo_profile(config_path, tf_profile_path):
    if not config_path or not tf_profile_path:
        raise ValueError("fast_livo requires mapping_config_file and tf_profile_path")
    config = yaml.safe_load(Path(config_path).read_text())["/**"]["ros__parameters"]
    tf = yaml.safe_load(Path(tf_profile_path).read_text())["omni_tf_manager"][
        "ros__parameters"
    ]
    if config["common"]["img_en"] != 1 or not config["vio"].get("lidar_timed_fusion"):
        raise ValueError("managed LIVO requires image-enabled complete-scan fusion")
    if config["imu"].get("acceleration_scale") != 1.0:
        raise ValueError("VBot LIVO requires the verified SI acceleration scale")
    edges = {}
    for key in tf["static_transforms"]:
        prefix = f"static_transform.{key}."
        r, p, y = tf[prefix + "rotation_rpy"]
        cr, sr, cp, sp, cy, sy = (
            math.cos(r),
            math.sin(r),
            math.cos(p),
            math.sin(p),
            math.cos(y),
            math.sin(y),
        )
        matrix = np.eye(4)
        matrix[:3, :3] = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        matrix[:3, 3] = tf[prefix + "translation"]
        edges[tf[prefix + "child_frame"]] = (tf[prefix + "parent_frame"], matrix)

    def base_pose(frame):
        transform = np.eye(4)
        seen = set()
        while frame != tf["frames.base"]:
            if frame in seen or frame not in edges:
                raise ValueError("TF profile has missing or cyclic sensor mounts")
            seen.add(frame)
            frame, step = edges[frame]
            transform = step @ transform
        return transform

    imu = base_pose(tf["frames.tracking"])
    lidar = base_pose(tf["frames.icp_sensor"])
    camera = base_pose(config["common"]["camera_frame"])
    ext = config["extrin_calib"]
    for expected, rotation, translation in [
        (np.linalg.inv(imu) @ lidar, "extrinsic_R", "extrinsic_T"),
        (np.linalg.inv(camera) @ lidar, "Rcl", "Pcl"),
    ]:
        if not np.allclose(
            expected[:3, :3], np.array(ext[rotation]).reshape(3, 3), atol=1e-5
        ) or not np.allclose(expected[:3, 3], ext[translation], atol=1e-5):
            raise ValueError(
                f"LIVO calibration disagrees with TF profile: {rotation}/{translation}"
            )
    return config
