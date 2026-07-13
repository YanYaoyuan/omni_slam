#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_dog3_env.sh"

cd "$SCRIPT_DIR"
if [ ! -x "$SCRIPT_DIR/.deps/venv/bin/colcon" ] || [ ! -f "$SCRIPT_DIR/.deps/root/usr/include/pcl-1.12/pcl/point_cloud.h" ]; then
  bash "$SCRIPT_DIR/bootstrap_dog3_deps.sh"
  source "$SCRIPT_DIR/setup_dog3_env.sh"
fi

colcon build --symlink-install \
  --packages-select fast_lio icp_relocalization \
  --cmake-args \
    -DOMNI_SLAM_DEPS_ROOT="$SCRIPT_DIR/.deps/root" \
    -DBUILD_SAC_IA_GICP=OFF \
    -DUSE_LIVOX=OFF

echo
echo "编译完成。使用："
echo "  bash 1_mapping.sh"
echo "  bash 2_localizing.sh"
