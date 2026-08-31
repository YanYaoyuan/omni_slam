#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf'^(\s*{re.escape(key)}\s*:\s*).*$',
                         flags=re.MULTILINE)
    replacement = rf'\1"{value}"'
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find yaml key: {key}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lidar-topic", required=True)
    parser.add_argument("--imu-topic", required=True)
    parser.add_argument("--prior-map-path", default=None)
    parser.add_argument("--map-file-path", default=None)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    text = replace_yaml_scalar(text, "lid_topic", args.lidar_topic)
    text = replace_yaml_scalar(text, "imu_topic", args.imu_topic)
    if args.prior_map_path is not None:
        text = replace_yaml_scalar(text, "prior_map_path", args.prior_map_path)
    if args.map_file_path is not None:
        text = replace_yaml_scalar(text, "map_file_path", args.map_file_path)
    Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
