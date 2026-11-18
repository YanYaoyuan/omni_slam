import pytest
import yaml
from omni_slam_manager.livo_profile import validate_livo_profile, LIVO_MAPPING_COMMAND
from omni_slam_manager.process_supervisor import expand_command


def write_profiles(tmp_path):
    tf = {
        "frames.base": "base",
        "frames.tracking": "imu",
        "frames.icp_sensor": "lidar",
        "static_transforms": ["i", "l", "c"],
    }
    for name, child, translation in [
        ("i", "imu", [0, 0, 0]),
        ("l", "lidar", [1, 0, 0]),
        ("c", "camera", [0, 0, 0]),
    ]:
        for k, v in dict(
            parent_frame="base",
            child_frame=child,
            translation=translation,
            rotation_rpy=[0, 0, 0],
        ).items():
            tf[f"static_transform.{name}.{k}"] = v
    cfg = {
        "common": {"img_en": 1, "camera_frame": "camera"},
        "vio": {"lidar_timed_fusion": True},
        "imu": {"acceleration_scale": 1.0},
        "extrin_calib": {
            "extrinsic_R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "extrinsic_T": [1, 0, 0],
            "Rcl": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "Pcl": [1, 0, 0],
        },
    }
    a = tmp_path / "livo.yaml"
    b = tmp_path / "tf.yaml"
    a.write_text(yaml.safe_dump({"/**": {"ros__parameters": cfg}}))
    b.write_text(yaml.safe_dump({"omni_tf_manager": {"ros__parameters": tf}}))
    return a, b


def test_matching_mounts_and_required_profiles(tmp_path):
    a, b = write_profiles(tmp_path)
    validate_livo_profile(a, b)
    with pytest.raises(ValueError):
        validate_livo_profile("", b)


def test_inconsistent_mount_cannot_start(tmp_path):
    a, b = write_profiles(tmp_path)
    d = yaml.safe_load(b.read_text())
    d["omni_tf_manager"]["ros__parameters"]["static_transform.i.rotation_rpy"] = [
        0,
        0,
        3.141592653589793,
    ]
    b.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="disagrees"):
        validate_livo_profile(a, b)


def test_managed_command_preserves_config_paths_with_spaces():
    values = {key: "frame" for key in ["odom_frame", "tracking_frame"]}
    values.update(
        mapping_config_file="/tmp/with spaces/calib.yaml",
        map_source_path="/tmp/map.pcd",
        runtime_directory="/tmp/run",
        use_sim_time="true",
    )
    args = expand_command(LIVO_MAPPING_COMMAND, values)
    assert "config:=/tmp/with spaces/calib.yaml" in args
    assert "use_sim_time:=true" in args
