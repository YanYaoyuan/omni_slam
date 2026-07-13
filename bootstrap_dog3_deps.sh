#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/.deps"
APT_DIR="$DEPS_DIR/apt"
ROOT_DIR="$DEPS_DIR/root"
VENV_DIR="$DEPS_DIR/venv"
SDK_ROOT="$DEPS_DIR/sdk_sysroot"
ROS_SDK_OVERLAY="$DEPS_DIR/ros_sdk_overlay"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "错误：该脚本仅用于 dog3 的 ARM64 系统" >&2
  exit 2
fi

mkdir -p "$APT_DIR/state/lists/partial" "$APT_DIR/cache/archives/partial" "$ROOT_DIR"
cat > "$APT_DIR/sources.list" <<'EOF'
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main universe multiverse restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe multiverse restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-security main universe multiverse restricted
EOF

APT_OPTIONS=(
  -o "Dir::State=$APT_DIR/state"
  -o "Dir::State::status=/var/lib/dpkg/status"
  -o "Dir::Cache=$APT_DIR/cache"
  -o "Dir::Etc::sourcelist=$APT_DIR/sources.list"
  -o "Dir::Etc::sourceparts=-"
  -o APT::Architecture=arm64
)

if [ ! -f "$APT_DIR/state/lists/ports.ubuntu.com_ubuntu-ports_dists_jammy_InRelease" ]; then
  apt-get "${APT_OPTIONS[@]}" update
fi

RUNTIME_PACKAGES=(
  libboost-date-time1.74.0
  libboost-filesystem1.74.0
  libboost-iostreams1.74.0
  libboost-serialization1.74.0
  libconsole-bridge1.0
  libflann1.9
  libfmt8
  liborocos-kdl1.5
  libpcl-common1.12
  libpcl-features1.12
  libpcl-filters1.12
  libpcl-kdtree1.12
  libpcl-octree1.12
  libpcl-registration1.12
  libpcl-sample-consensus1.12
  libpcl-search1.12
  libqhull-r8.0
  libqhull8.0
  libqhullcpp8.0
  libspdlog1
)

apt-get "${APT_OPTIONS[@]}" --download-only --no-install-recommends install "${RUNTIME_PACKAGES[@]}"

pushd "$APT_DIR/cache/archives" >/dev/null
HEADER_PACKAGES=(
  libboost1.74-dev
  libconsole-bridge-dev
  libeigen3-dev
  libflann-dev
  libfmt-dev
  liblz4-dev
  liborocos-kdl-dev
  libpcl-dev
  libqhull-dev
  libspdlog-dev
)
for package in "${HEADER_PACKAGES[@]}"; do
  if ! compgen -G "${package}_*.deb" >/dev/null; then
    apt-get "${APT_OPTIONS[@]}" download "$package"
  fi
done
for package_file in ./*.deb; do
  dpkg-deb -x "$package_file" "$ROOT_DIR"
done
popd >/dev/null

mkdir -p "$SDK_ROOT/usr/include" "$SDK_ROOT/usr/lib/aarch64-linux-gnu" "$ROS_SDK_OVERLAY/share"
ln -sfn /app/opt/ros/humble/include "$ROS_SDK_OVERLAY/include"
ln -sfn /app/opt/ros/humble/lib "$ROS_SDK_OVERLAY/lib"
ln -sfn "$ROOT_DIR/usr/include/eigen3" "$SDK_ROOT/usr/include/eigen3"
ln -sfn /usr/include/aarch64-linux-gnu "$SDK_ROOT/usr/include/aarch64-linux-gnu"
ln -sfn /usr/lib/aarch64-linux-gnu/libpython3.10.so.1.0 "$SDK_ROOT/usr/lib/aarch64-linux-gnu/libpython3.10.so"
ln -sfn /usr/lib/aarch64-linux-gnu/libcurl.so.4 "$SDK_ROOT/usr/lib/aarch64-linux-gnu/libcurl.so"
ln -sfn /app/opt/ros/humble/lib/libtinyxml2.so.9 "$SDK_ROOT/usr/lib/aarch64-linux-gnu/libtinyxml2.so"

while IFS= read -r cmake_file; do
  package_dir="$(dirname "$(dirname "$cmake_file")")"
  package_name="$(basename "$package_dir")"
  mkdir -p "$ROS_SDK_OVERLAY/share/$package_name"
  rm -rf "$ROS_SDK_OVERLAY/share/$package_name/cmake"
  cp -a "$package_dir/cmake" "$ROS_SDK_OVERLAY/share/$package_name/cmake"
done < <(grep -Rsl '/sysroot/usr' /app/opt/ros/humble/share/*/cmake 2>/dev/null | sort -u)

while IFS= read -r cmake_file; do
  sed -i "s#/sysroot/usr#$SDK_ROOT/usr#g" "$cmake_file"
done < <(grep -Rsl '/sysroot/usr' "$ROS_SDK_OVERLAY/share" 2>/dev/null || true)

if [ ! -x "$VENV_DIR/bin/python3" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python3" -m pip install --disable-pip-version-check --upgrade pip
"$VENV_DIR/bin/python3" -m pip install --disable-pip-version-check \
  cmake==3.28.4 empy==3.3.4 lark==1.1.2 numpy==1.26.4 colcon-common-extensions

echo "dog3 私有依赖已准备完成：$DEPS_DIR"
