#!/usr/bin/env bash
set -eo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLAM_ROOT="$(cd "${TOOL_DIR}/../.." && pwd)"

if [[ -f "${SLAM_ROOT}/setup_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SLAM_ROOT}/setup_env.sh"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f /app/opt/ros/humble/local_setup.bash ]]; then
  # shellcheck disable=SC1091
  source /app/opt/ros/humble/local_setup.bash
else
  echo "错误：未找到 ROS 2 Humble 环境" >&2
  exit 2
fi

if [[ -f "${SLAM_ROOT}/install/local_setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${SLAM_ROOT}/install/local_setup.bash"
fi

exec python3 "${TOOL_DIR}/publish_path.py" "$@"
