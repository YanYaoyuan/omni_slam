#!/usr/bin/env python3
"""Derive matching LIVO/TF profiles from a recorded rig, without changing defaults."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--livo-config", type=Path, required=True)
    ap.add_argument("--tf-template", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output exists")
    e = next(e for e in json.loads(args.inventory.read_text()) if e["name"] == args.bag)
    cfg = yaml.safe_load(args.livo_config.read_text())
    p = cfg["/**"]["ros__parameters"]
    p["use_sim_time"] = True
    p["common"].update(
        lid_topic="/omni/sensors/lidar/points",
        imu_topic="/omni/sensors/imu/data",
        img_topic="/omni/sensors/camera/image_rect",
    )
    tf = yaml.safe_load(args.tf_template.read_text())
    t = tf["omni_tf_manager"]["ros__parameters"]
    t["profile_name"] = args.bag + "_recorded"
    t["calibration_id"] = args.bag + "_CameraInfo_and_static_TF"
    t["use_sim_time"] = True
    # The replay publishes canonical payloads directly; never relay a second copy.
    t.pop("sensor_relays", None)  # An empty YAML sequence has no ROS parameter type.
    for key in list(t):
        if key.startswith("sensor_relay.") or key.startswith("static_transform."):
            del t[key]
    transforms = e["calibration"]["transforms"]
    bi = np.array(transforms["lidar_imu"])
    bl = np.array(transforms["lidar"])
    bc = np.array(transforms["stereo_left"])
    mounts = [
        ("imu", "omni_base_link", "omni_imu_link", bi),
        ("lidar", "omni_imu_link", "omni_lidar_link", np.linalg.inv(bi) @ bl),
        (
            "camera",
            "omni_imu_link",
            "omni_rgb_camera_optical_frame",
            np.linalg.inv(bi) @ bc,
        ),
    ]
    t["static_transforms"] = [name + "_mount" for name, *_ in mounts]
    for name, parent, child, matrix in mounts:
        prefix = f"static_transform.{name}_mount."
        t.update(
            {
                prefix + "role": name,
                prefix + "parent_frame": parent,
                prefix + "child_frame": child,
                prefix + "translation": matrix[:3, 3].tolist(),
                prefix + "rotation_rpy": Rotation.from_matrix(matrix[:3, :3])
                .as_euler("xyz")
                .tolist(),
            }
        )
    args.output.mkdir(parents=True)
    (args.output / "livo.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (args.output / "tf.yaml").write_text(yaml.safe_dump(tf, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
