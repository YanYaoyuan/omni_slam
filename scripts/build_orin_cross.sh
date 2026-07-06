#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SYSROOT=${ORIN_SYSROOT:-$HOME/sysroots/orin}
BUILD_BASE=${ORIN_BUILD_BASE:-$ROOT_DIR/build_orin}
INSTALL_BASE=${ORIN_INSTALL_BASE:-$ROOT_DIR/install_orin}
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
TOOLCHAIN=${ORIN_TOOLCHAIN_FILE:-$ROOT_DIR/toolchains/orin-aarch64.cmake}
TOOLCHAIN_ROOT=${ORIN_TOOLCHAIN_ROOT:-}
LIVOX_PREFIX_IN_SYSROOT=${LIVOX_PREFIX_IN_SYSROOT:-/home/user/ws_livox/install}

die() { echo "[build_orin_cross] ERROR: $*" >&2; exit 1; }
check_path() { [[ -e "$1" ]] || die "$2: $1"; }

check_path "$SYSROOT" "missing ORIN_SYSROOT; run scripts/sync_orin_sysroot.sh user@orin first"
check_path "$SYSROOT/opt/ros/$ROS_DISTRO_TARGET" "missing target ROS 2 in sysroot"
check_path "$SYSROOT/usr/include/pcl-1.12" "missing target PCL headers in sysroot"
check_path "$SYSROOT$LIVOX_PREFIX_IN_SYSROOT" "missing target Livox install in sysroot"
check_path "$TOOLCHAIN" "missing CMake toolchain file"
if [[ -n "$TOOLCHAIN_ROOT" ]]; then
  check_path "$TOOLCHAIN_ROOT/bin/aarch64-buildroot-linux-gnu-g++" "missing Buildroot aarch64 g++ in ORIN_TOOLCHAIN_ROOT"
else
  command -v aarch64-linux-gnu-g++ >/dev/null || die "missing aarch64-linux-gnu-g++"
fi
command -v colcon >/dev/null || die "missing colcon"

export ORIN_SYSROOT="$SYSROOT"
export ORIN_STAGING_PREFIX="$INSTALL_BASE"

source "/opt/ros/$ROS_DISTRO_TARGET/setup.bash"

CMAKE_PREFIX_PATH_TARGET="$SYSROOT/opt/ros/$ROS_DISTRO_TARGET;$SYSROOT$LIVOX_PREFIX_IN_SYSROOT;$SYSROOT/usr"

echo "[build_orin_cross] root: $ROOT_DIR"
echo "[build_orin_cross] sysroot: $SYSROOT"
if [[ -n "$TOOLCHAIN_ROOT" ]]; then
  echo "[build_orin_cross] toolchain root: $TOOLCHAIN_ROOT"
fi
echo "[build_orin_cross] install: $INSTALL_BASE"

cd "$ROOT_DIR"
colcon build \
  --merge-install \
  --build-base "$BUILD_BASE" \
  --install-base "$INSTALL_BASE" \
  --packages-select fast_lio icp_relocalization \
  --cmake-args \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_TARGET"

file "$INSTALL_BASE/lib/fast_lio/fastlio_mapping" || true
file "$INSTALL_BASE/lib/icp_relocalization/icp_node" || true

echo "[build_orin_cross] done"
