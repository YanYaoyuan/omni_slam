#!/usr/bin/env bash
#
# Cross-compile Omni SLAM with the downloaded Orin NX sysroot and package it.
#
# Default sysroot:
#   /home/user/jetson/orin-nx/sysroot
#
# Usage:
#   ./scripts/build_and_package_orin_nx_sysroot.sh
#
# Common overrides:
#   ORIN_SYSROOT=/path/to/sysroot ./scripts/build_and_package_orin_nx_sysroot.sh
#   MAP_PATH=/path/to/map.pcd ./scripts/build_and_package_orin_nx_sysroot.sh
#   VERSION=test_001 ./scripts/build_and_package_orin_nx_sysroot.sh
#
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SYSROOT=${ORIN_SYSROOT:-/home/user/jetson/orin-nx/sysroot}
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
BUILD_BASE=${ORIN_BUILD_BASE:-$ROOT_DIR/build_orin_nx_sysroot}
INSTALL_BASE=${ORIN_INSTALL_BASE:-$ROOT_DIR/install_orin_nx_sysroot}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/dist}
VERSION=${VERSION:-$(date +%Y%m%d_%H%M%S)}
MAP_PATH=${MAP_PATH:-$ROOT_DIR/FAST_LIO/PCD/omni_dog_map.pcd}
PACKAGE_MODE=${PACKAGE_MODE:-flat}
PATCH_SYSROOT_SCRIPT=${PATCH_SYSROOT_SCRIPT:-/home/user/robot/SCAN-Planner/tools/cross/patch_orin_sysroot_cmake.sh}

die() { echo "[orin_nx_sysroot] ERROR: $*" >&2; exit 1; }
log() { echo "[orin_nx_sysroot] $*"; }

[[ -d "$SYSROOT" ]] || die "missing sysroot: $SYSROOT"
[[ -f "$SYSROOT/opt/ros/$ROS_DISTRO_TARGET/setup.bash" ]] || die "missing target ROS setup: $SYSROOT/opt/ros/$ROS_DISTRO_TARGET/setup.bash"
[[ -d "$SYSROOT/usr/include/pcl-1.12" ]] || die "missing target PCL headers: $SYSROOT/usr/include/pcl-1.12"
[[ -x "$SYSROOT/usr/bin/aarch64-linux-gnu-g++" ]] || die "missing target compiler: $SYSROOT/usr/bin/aarch64-linux-gnu-g++"

if command -v qemu-aarch64-static >/dev/null 2>&1; then
  :
elif command -v qemu-aarch64 >/dev/null 2>&1; then
  :
else
  die "missing qemu-aarch64/qemu-aarch64-static; required when using the compiler copied inside the sysroot"
fi

if [[ -x "$PATCH_SYSROOT_SCRIPT" ]]; then
  log "patch sysroot CMake exports: $PATCH_SYSROOT_SCRIPT"
  "$PATCH_SYSROOT_SCRIPT" "$SYSROOT"
else
  log "warning: sysroot patch script not found, skip: $PATCH_SYSROOT_SCRIPT"
fi

log "cross build"
log "  source:  $ROOT_DIR"
log "  sysroot: $SYSROOT"
log "  build:   $BUILD_BASE"
log "  install: $INSTALL_BASE"

ORIN_SYSROOT="$SYSROOT" \
ORIN_BUILD_BASE="$BUILD_BASE" \
ORIN_INSTALL_BASE="$INSTALL_BASE" \
ROS_DISTRO_TARGET="$ROS_DISTRO_TARGET" \
ORIN_USE_SYSROOT_TOOLCHAIN="${ORIN_USE_SYSROOT_TOOLCHAIN:-1}" \
"$ROOT_DIR/scripts/build_orin_cross.sh"

export ORIN_INSTALL_BASE="$INSTALL_BASE"
export OUT_DIR
export VERSION
export MAP_PATH

case "$PACKAGE_MODE" in
  flat)
    PKG_SCRIPT="$ROOT_DIR/scripts/package_orin_flat_runtime.sh"
    ;;
  install)
    PKG_SCRIPT="$ROOT_DIR/scripts/package_orin_runtime.sh"
    ;;
  *)
    die "unsupported PACKAGE_MODE=$PACKAGE_MODE; expected flat or install"
    ;;
esac

log "package with $(basename "$PKG_SCRIPT")"
PKG_OUTPUT=$("$PKG_SCRIPT")
printf '%s\n' "$PKG_OUTPUT"
TARBALL=$(printf '%s\n' "$PKG_OUTPUT" | tail -n 1)

if [[ -f "$TARBALL" ]]; then
  log "sha256:"
  sha256sum "$TARBALL"
fi

log "verify target binaries:"
file "$INSTALL_BASE/lib/fast_lio/fastlio_mapping"
file "$INSTALL_BASE/lib/icp_relocalization/icp_node"

log "done"
