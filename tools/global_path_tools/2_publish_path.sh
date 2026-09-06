#!/usr/bin/env bash
set -eo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLAM_ROOT="$(cd "${TOOL_DIR}/../.." && pwd)"

# 与录制入口保持一致：环境 hook 运行期间关闭 nounset，完成后恢复调用方状态。
NOUNSET_WAS_ENABLED=0
if [[ "$-" == *u* ]]; then
  NOUNSET_WAS_ENABLED=1
fi
set +u

if [[ -f "${SLAM_ROOT}/setup_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SLAM_ROOT}/setup_env.sh"
  set +u
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
  set +u
  # shellcheck disable=SC1091
  source "${SLAM_ROOT}/install/local_setup.bash"
fi

if [[ "$NOUNSET_WAS_ENABLED" -eq 1 ]]; then
  set -u
else
  set +u
fi

exec python3 "${TOOL_DIR}/publish_path.py" "$@"
