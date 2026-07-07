#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SYSROOT=${ORIN_SYSROOT:-$HOME/sysroots/orin}
BUILD_BASE=${ORIN_BUILD_BASE:-$ROOT_DIR/build_orin}
INSTALL_BASE=${ORIN_INSTALL_BASE:-$ROOT_DIR/install_orin}
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
TOOLCHAIN=${ORIN_TOOLCHAIN_FILE:-$ROOT_DIR/toolchains/orin-aarch64.cmake}
TOOLCHAIN_ROOT=${ORIN_TOOLCHAIN_ROOT:-}
USE_SYSROOT_TOOLCHAIN=${ORIN_USE_SYSROOT_TOOLCHAIN:-1}
LIVOX_PREFIX_IN_SYSROOT=${LIVOX_PREFIX_IN_SYSROOT:-/home/user/ws_livox/install}

die() { echo "[build_orin_cross] ERROR: $*" >&2; exit 1; }
check_path() { [[ -e "$1" ]] || die "$2: $1"; }

check_path "$SYSROOT" "missing ORIN_SYSROOT; run scripts/sync_orin_sysroot.sh user@orin first"
check_path "$SYSROOT/opt/ros/$ROS_DISTRO_TARGET" "missing target ROS 2 in sysroot"
check_path "$SYSROOT/usr/include/pcl-1.12" "missing target PCL headers in sysroot"
check_path "$SYSROOT$LIVOX_PREFIX_IN_SYSROOT" "missing target Livox install in sysroot"
check_path "$TOOLCHAIN" "missing CMake toolchain file"
if [[ "$USE_SYSROOT_TOOLCHAIN" == "1" ]]; then
  check_path "$SYSROOT/usr/bin/aarch64-linux-gnu-g++" "missing target aarch64 g++ in sysroot"
  command -v qemu-aarch64-static >/dev/null || command -v qemu-aarch64 >/dev/null || die "missing qemu-aarch64 to run target compiler"
elif [[ -n "$TOOLCHAIN_ROOT" ]]; then
  check_path "$TOOLCHAIN_ROOT/bin/aarch64-buildroot-linux-gnu-g++" "missing Buildroot aarch64 g++ in ORIN_TOOLCHAIN_ROOT"
else
  command -v aarch64-linux-gnu-g++ >/dev/null || die "missing aarch64-linux-gnu-g++"
fi
command -v colcon >/dev/null || die "missing colcon"

export ORIN_SYSROOT="$SYSROOT"
export ORIN_STAGING_PREFIX="$INSTALL_BASE"
export ORIN_USE_SYSROOT_TOOLCHAIN="$USE_SYSROOT_TOOLCHAIN"
if [[ "$USE_SYSROOT_TOOLCHAIN" == "1" ]]; then
  export QEMU_LD_PREFIX="$SYSROOT"
  export PATH="$SYSROOT/usr/bin:$PATH"
fi

set +u
source "/opt/ros/$ROS_DISTRO_TARGET/setup.bash"
set -u

CMAKE_PREFIX_PATH_TARGET="$INSTALL_BASE;$SYSROOT/opt/ros/$ROS_DISTRO_TARGET;$SYSROOT$LIVOX_PREFIX_IN_SYSROOT;$SYSROOT/usr"
MAKE_WRAPPER="$BUILD_BASE/host-make-wrapper.sh"
mkdir -p "$BUILD_BASE"
cat > "$MAKE_WRAPPER" <<'MAKE'
#!/usr/bin/env bash
exec /usr/bin/make MAKE="$0" \
  -o /usr/lib/aarch64-linux-gnu/libpython3.10.so \
  -o /usr/lib/aarch64-linux-gnu/libpcl_common.so \
  -o /usr/lib/libOpenNI.so \
  -o /usr/lib/aarch64-linux-gnu/libOpenNI2.so \
  "$@"
MAKE
chmod +x "$MAKE_WRAPPER"
export ORIN_MAKE_PROGRAM="$MAKE_WRAPPER"

PACKAGE_PATHS=(FAST_LIO icp_relocalization)
PACKAGE_SELECT=(fast_lio icp_relocalization)
if [[ ! -f "$SYSROOT$LIVOX_PREFIX_IN_SYSROOT/share/livox_ros_driver2/cmake/livox_ros_driver2Config.cmake" && ! -f "$SYSROOT/opt/ros/$ROS_DISTRO_TARGET/share/livox_ros_driver2/cmake/livox_ros_driver2Config.cmake" ]]; then
  LIVOX_STUB_SRC="$BUILD_BASE/livox_ros_driver2_stub_src/livox_ros_driver2"
  echo "[build_orin_cross] target livox_ros_driver2 not found; generating message-only stub: $LIVOX_STUB_SRC"
  mkdir -p "$LIVOX_STUB_SRC/msg"
  cat > "$LIVOX_STUB_SRC/package.xml" <<'PKG'
<?xml version="1.0"?>
<package format="3">
  <name>livox_ros_driver2</name>
  <version>0.0.0</version>
  <description>Message-only Livox compatibility package for cross-compilation.</description>
  <maintainer email="ci@example.com">CI</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>
  <depend>std_msgs</depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
PKG
  cat > "$LIVOX_STUB_SRC/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.8)
project(livox_ros_driver2)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/CustomPoint.msg
  msg/CustomMsg.msg
  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
CMAKE
  cat > "$LIVOX_STUB_SRC/msg/CustomPoint.msg" <<'MSG'
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
MSG
  cat > "$LIVOX_STUB_SRC/msg/CustomMsg.msg" <<'MSG'
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
CustomPoint[] points
MSG
  PACKAGE_PATHS=("$LIVOX_STUB_SRC" "${PACKAGE_PATHS[@]}")
  PACKAGE_SELECT=(livox_ros_driver2 "${PACKAGE_SELECT[@]}")
fi

echo "[build_orin_cross] root: $ROOT_DIR"
echo "[build_orin_cross] sysroot: $SYSROOT"
if [[ "$USE_SYSROOT_TOOLCHAIN" == "1" ]]; then
  echo "[build_orin_cross] toolchain: $SYSROOT/usr/bin/aarch64-linux-gnu-g++"
elif [[ -n "$TOOLCHAIN_ROOT" ]]; then
  echo "[build_orin_cross] toolchain root: $TOOLCHAIN_ROOT"
fi
echo "[build_orin_cross] install: $INSTALL_BASE"

cd "$ROOT_DIR"
colcon build \
  --merge-install \
  --build-base "$BUILD_BASE" \
  --install-base "$INSTALL_BASE" \
  --paths "${PACKAGE_PATHS[@]}" \
  --packages-select "${PACKAGE_SELECT[@]}" \
  --cmake-args \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_MAKE_PROGRAM="$MAKE_WRAPPER" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LINK_DEPENDS_NO_SHARED=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_rosidl_generator_py=TRUE \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPython3_LIBRARY="$SYSROOT/usr/lib/aarch64-linux-gnu/libpython3.10.so" \
    -DPython3_INCLUDE_DIR="$SYSROOT/usr/include/python3.10" \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_LIBRARY="$SYSROOT/usr/lib/aarch64-linux-gnu/libpython3.10.so" \
    -DPYTHON_LIBRARIES="$SYSROOT/usr/lib/aarch64-linux-gnu/libpython3.10.so" \
    -DPYTHON_INCLUDE_DIR="$SYSROOT/usr/include/python3.10" \
    -DPYTHON_INCLUDE_DIRS="$SYSROOT/usr/include/python3.10" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_TARGET"

file "$INSTALL_BASE/lib/fast_lio/fastlio_mapping" || true
file "$INSTALL_BASE/lib/icp_relocalization/icp_node" || true

echo "[build_orin_cross] done"
