"""Read the core frame contract owned by an omni_tf_manager profile."""

from pathlib import Path
from typing import Dict

import yaml


DEFAULT_FRAMES = {
    "map": "omni_map",
    "odom": "omni_odom",
    "base": "omni_base_link",
    "tracking": "omni_imu_link",
    "icp_sensor": "omni_lidar_link",
}


def load_tf_frames(profile_path: str) -> Dict[str, str]:
    """Return core frames from a manager YAML, or Omni defaults for an empty path."""
    if not profile_path.strip():
        return dict(DEFAULT_FRAMES)
    path = Path(profile_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"TF profile does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        parameters = document["omni_tf_manager"]["ros__parameters"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid omni_tf_manager profile {path}: {error}") from error

    frames = {}
    for role in DEFAULT_FRAMES:
        value = parameters.get(f"frames.{role}")
        if not isinstance(value, str) or not value.startswith("omni_"):
            raise ValueError(
                f"{path}: frames.{role} must be a string beginning with 'omni_'"
            )
        frames[role] = value
    if len(set(frames.values())) != len(frames):
        raise ValueError(f"{path}: core TF frame names must be unique")
    return frames
