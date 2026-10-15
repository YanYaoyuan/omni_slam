#!/usr/bin/env bash

# Reproducible RDK S100 cross build shared by GitHub Actions and local runs.
# All generated source, sysroot and colcon output live below S100_WORK_ROOT.

set -euo pipefail

report_error() {
  local exit_code=$?
  printf '[s100-cross] ERROR: line %s: %s (exit %s)\n' \
    "$1" "$2" "${exit_code}" >&2
  exit "${exit_code}"
}

trap 'report_error "${LINENO}" "${BASH_COMMAND}"' ERR

SOURCE_ROOT=${S100_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TF_MANAGER_SOURCE=${OMNI_TF_MANAGER_SOURCE:-${SOURCE_ROOT}/../omni_tf_manager}
ROBOT_INTERFACES_SOURCE=${OMNI_ROBOT_INTERFACES_SOURCE:-${SOURCE_ROOT}/../omni_robot_interfaces}
WORK_ROOT=${S100_WORK_ROOT:-${RUNNER_TEMP:-/tmp}/s100-cross}
S100_IMAGE=${S100_IMAGE:-pc_tros_ubuntu22.04:v1.0.0}
ROBOT_DEV_CONFIG_REF=${ROBOT_DEV_CONFIG_REF:-f44b1ad2575b34c189fd470de1db9c5ad48b6886}
S100_SYSROOT_REF=${S100_SYSROOT_REF:-de9fa286f71a72d24c349477cd59f41cc2cc3d8f}
ROBOT_DEV_CONFIG_URL=${ROBOT_DEV_CONFIG_URL:-https://github.com/D-Robotics/robot_dev_config.git}
S100_SYSROOT_URL=${S100_SYSROOT_URL:-https://github.com/D-Robotics/sysroot_docker.git}
TROS_ARM_BUILD_URL=${TROS_ARM_BUILD_URL:-https://github.com/D-Robotics/tros_arm_build.git}
S100_CLEAN=${S100_CLEAN:-1}

log() {
  printf '[s100-cross] %s\n' "$*"
}

die() {
  printf '[s100-cross] ERROR: %s\n' "$*" >&2
  exit 1
}

for command_name in docker git python3 rsync; do
  command -v "${command_name}" >/dev/null 2>&1 || die "missing command: ${command_name}"
done

SOURCE_ROOT=$(realpath -m "${SOURCE_ROOT}")
TF_MANAGER_SOURCE=$(realpath -m "${TF_MANAGER_SOURCE}")
ROBOT_INTERFACES_SOURCE=$(realpath -m "${ROBOT_INTERFACES_SOURCE}")
WORK_ROOT=$(realpath -m "${WORK_ROOT}")

[[ -f "${TF_MANAGER_SOURCE}/package.xml" ]] || \
  die "invalid OMNI_TF_MANAGER_SOURCE: ${TF_MANAGER_SOURCE}"
[[ -f "${ROBOT_INTERFACES_SOURCE}/package.xml" ]] || \
  die "invalid OMNI_ROBOT_INTERFACES_SOURCE: ${ROBOT_INTERFACES_SOURCE}"

case "${WORK_ROOT}" in
  /|/data|/home|"${SOURCE_ROOT}")
    die "unsafe S100_WORK_ROOT: ${WORK_ROOT}"
    ;;
esac

TROS_ROOT="${WORK_ROOT}/cc_ws/tros_ws"
ROBOT_DEV_CONFIG_DIR="${TROS_ROOT}/robot_dev_config"

if [[ "${S100_CLEAN}" == "1" ]]; then
  log "cleaning generated workspace: ${WORK_ROOT}/cc_ws"
  rm -rf "${WORK_ROOT}/cc_ws"
fi

mkdir -p \
  "${TROS_ROOT}/src/omni_slam" \
  "${TROS_ROOT}/src/omni_tf_manager" \
  "${TROS_ROOT}/src/omni_robot_interfaces"

if [[ ! -d "${ROBOT_DEV_CONFIG_DIR}/.git" ]]; then
  log "cloning robot_dev_config at ${ROBOT_DEV_CONFIG_REF}"
  git clone --filter=blob:none \
    "${ROBOT_DEV_CONFIG_URL}" \
    "${ROBOT_DEV_CONFIG_DIR}"
else
  log "reusing robot_dev_config checkout at ${ROBOT_DEV_CONFIG_REF}"
  git -C "${ROBOT_DEV_CONFIG_DIR}" remote set-url origin \
    "${ROBOT_DEV_CONFIG_URL}"
  if ! git -C "${ROBOT_DEV_CONFIG_DIR}" cat-file -e \
      "${ROBOT_DEV_CONFIG_REF}^{commit}" 2>/dev/null; then
    git -C "${ROBOT_DEV_CONFIG_DIR}" fetch --no-tags origin \
      "${ROBOT_DEV_CONFIG_REF}"
  fi
fi
git -C "${ROBOT_DEV_CONFIG_DIR}" checkout --detach "${ROBOT_DEV_CONFIG_REF}"

log "copying Omni SLAM source into the cross workspace"
rsync -a --delete \
  --exclude='/.git/' \
  --exclude='/build/' \
  --exclude='/build_orin/' \
  --exclude='/install/' \
  --exclude='/install_orin/' \
  --exclude='/log/' \
  --exclude='/dist/' \
  --exclude='/omni_tf_manager_source/' \
  --exclude='/omni_robot_interfaces_source/' \
  "${SOURCE_ROOT}/" \
  "${TROS_ROOT}/src/omni_slam/"

log "copying Omni TF Manager source into the cross workspace"
rsync -a --delete \
  --exclude='/.git/' \
  --exclude='/build/' \
  --exclude='/install/' \
  --exclude='/log/' \
  "${TF_MANAGER_SOURCE}/" \
  "${TROS_ROOT}/src/omni_tf_manager/"

log "copying Omni Robot Interfaces source into the cross workspace"
rsync -a --delete \
  --exclude='/.git/' \
  --exclude='/build/' \
  --exclude='/install/' \
  --exclude='/log/' \
  "${ROBOT_INTERFACES_SOURCE}/" \
  "${TROS_ROOT}/src/omni_robot_interfaces/"

test -f "${TROS_ROOT}/src/omni_slam/FAST_LIO/include/IKFoM_toolkit/mtk/build_manifold.hpp"

log "starting toolchain container"
docker run --rm -i \
  --network host \
  -e "S100_SYSROOT_REF=${S100_SYSROOT_REF}" \
  -e "S100_SYSROOT_URL=${S100_SYSROOT_URL}" \
  -e "TROS_ARM_BUILD_URL=${TROS_ARM_BUILD_URL}" \
  -e "HTTP_PROXY=${S100_CONTAINER_HTTP_PROXY:-${HTTP_PROXY:-}}" \
  -e "HTTPS_PROXY=${S100_CONTAINER_HTTPS_PROXY:-${HTTPS_PROXY:-}}" \
  -e "NO_PROXY=${NO_PROXY:-}" \
  -v "${WORK_ROOT}:/mnt/s100-cross" \
  -w /mnt/s100-cross/cc_ws/tros_ws \
  "${S100_IMAGE}" \
  bash -s <<'CONTAINER'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
if [[ -f /mnt/s100-cross/.codeup-git-credentials ]]; then
  git config --global credential.helper \
    'store --file=/mnt/s100-cross/.codeup-git-credentials'
fi
report_container_error() {
  local exit_code=$?
  printf '[s100-container] ERROR: line %s: %s (exit %s)\n' \
    "$1" "$2" "${exit_code}" >&2
  exit "${exit_code}"
}

trap 'report_container_error "${LINENO}" "${BASH_COMMAND}"' ERR

echo "=== Check toolchain and workspace ==="
for required_file in \
  robot_dev_config/build.sh \
  robot_dev_config/s100_build.sh \
  robot_dev_config/aarch64_toolchainfile.cmake \
  robot_dev_config/ros2.repos; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: missing required toolchain file: ${required_file}" >&2
    exit 1
  fi
done

shallow_checkout() {
  local url=$1
  local ref=$2
  local target=$3
  local attempt

  if [[ ! -d "${target}/.git" ]]; then
    mkdir -p "${target}"
    git -C "${target}" init --quiet
    git -C "${target}" remote add origin "${url}"
  else
    git -C "${target}" remote set-url origin "${url}"
  fi

  # Force HTTP/1.1 because the local proxy corrupted a long HTTP/2 TLS stream.
  # Low-speed detection prevents an otherwise healthy process from hanging
  # forever if the direct or proxied connection stops delivering bytes.
  for attempt in 1 2 3; do
    if git -C "${target}" \
      -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1024 \
      -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --depth=1 origin "${ref}"; then
      break
    fi
    if (( attempt == 3 )); then
      echo "ERROR: failed to fetch ${url} at ${ref} after ${attempt} attempts" >&2
      return 1
    fi
    echo "Retrying ${url} (${attempt}/3)" >&2
  done
  git -C "${target}" checkout --quiet --detach FETCH_HEAD
}

checkout_sysroot_in_batches() {
  local ref=$1
  local cache=/mnt/s100-cross/cache/sysroot_docker-${ref}
  local complete_marker=${cache}/.usr_s100-complete
  local marker_dir=${cache}/.git/s100-completed-chunks
  local chunk_dir
  local oid_dir
  local chunk
  local chunk_name
  local export_index
  local export_root
  local oid_chunk
  local oid_part
  local oid_part_name
  local attempt
  local completed=0
  local missing_count
  local total=0

  mkdir -p "${cache}"
  if [[ ! -d "${cache}/.git" ]]; then
    git -C "${cache}" init --quiet
    git -C "${cache}" remote add origin "${S100_SYSROOT_URL}"
  else
    git -C "${cache}" remote set-url origin "${S100_SYSROOT_URL}"
  fi

  git -C "${cache}" config http.version HTTP/1.1
  git -C "${cache}" config http.lowSpeedLimit 1024
  git -C "${cache}" config http.lowSpeedTime 60
  git -C "${cache}" config gc.auto 0
  # An interrupted container can leave this transient lock behind. Builds are
  # serialized by test_s100_local.sh, so no other process can own it here.
  rm -f "${cache}/.git/index.lock"

  if ! git -C "${cache}" cat-file -e "${ref}^{commit}" 2>/dev/null; then
    for attempt in 1 2 3; do
      if git -C "${cache}" fetch \
        --progress --no-tags --depth=1 --filter=blob:none origin "${ref}"; then
        break
      fi
      if (( attempt == 3 )); then
        echo "ERROR: failed to fetch sysroot metadata after ${attempt} attempts" >&2
        return 1
      fi
      echo "Retrying sysroot metadata (${attempt}/3)" >&2
    done
  fi

  git -C "${cache}" config remote.origin.promisor true
  git -C "${cache}" config remote.origin.partialclonefilter blob:none
  git -C "${cache}" update-ref refs/heads/s100-cache "${ref}"
  git -C "${cache}" symbolic-ref HEAD refs/heads/s100-cache

  if [[ -f "${complete_marker}" ]] && \
    [[ $(<"${complete_marker}") == "${ref}" ]]; then
    echo "Reusing complete sysroot cache: ${cache}"
  else
    mkdir -p "${marker_dir}"
    chunk_dir=$(mktemp -d /tmp/s100-sysroot-chunks.XXXXXX)
    oid_dir=$(mktemp -d /tmp/s100-sysroot-oids.XXXXXX)

    git -C "${cache}" ls-tree -r --name-only "${ref}" usr_s100 \
      | sed 's|^|/|' \
      | split -l 2000 -d -a 4 - "${chunk_dir}/chunk-"
    git -C "${cache}" ls-tree -r "${ref}" usr_s100 \
      | awk '{print $3}' \
      | split -l 2000 -d -a 4 - "${oid_dir}/chunk-"

    total=$(find "${chunk_dir}" -maxdepth 1 -type f -name 'chunk-*' | wc -l)

    for chunk in "${chunk_dir}"/chunk-*; do
      chunk_name=$(basename "${chunk}")
      if [[ -f "${marker_dir}/${chunk_name}" ]]; then
        completed=$((completed + 1))
        continue
      fi

      # A single checkout can request hundreds of MiB and is prone to being
      # dropped by proxies. Ask Git's partial-clone promisor for small groups
      # of blobs first, and leave per-group markers so a retry resumes.
      oid_chunk="${oid_dir}/${chunk_name}"
      split -l 250 -d -a 4 "${oid_chunk}" "${oid_chunk}.part-"
      for oid_part in "${oid_chunk}".part-*; do
        oid_part_name=$(basename "${oid_part}")
        if [[ -f "${marker_dir}/${oid_part_name}" ]]; then
          continue
        fi

        for attempt in 1 2 3; do
          if git -C "${cache}" \
              -c fetch.negotiationAlgorithm=noop \
              -c http.version=HTTP/1.1 \
              fetch origin \
              --no-tags \
              --no-write-fetch-head \
              --recurse-submodules=no \
              --filter=blob:none \
              --stdin < "${oid_part}"; then
            break
          fi
          if (( attempt == 3 )); then
            echo "ERROR: sysroot blobs ${oid_part_name} failed after ${attempt} attempts" >&2
            return 1
          fi
          echo "Retrying sysroot blobs ${oid_part_name} (${attempt}/3)" >&2
        done
        touch "${marker_dir}/${oid_part_name}"
      done

      touch "${marker_dir}/${chunk_name}"
      completed=$((completed + 1))
      echo "sysroot chunks: ${completed}/${total}"
    done

    missing_count=$(git -C "${cache}" rev-list \
      --objects --missing=print "${ref}" -- usr_s100 \
      | grep -c '^?' || true)
    if (( missing_count != 0 )); then
      echo "ERROR: sysroot object cache is missing ${missing_count} blobs" >&2
      return 1
    fi
    printf '%s\n' "${ref}" > "${complete_marker}"
  fi

  test "$(git -C "${cache}" rev-parse HEAD)" = "${ref}"
  mkdir -p ../sysroot_docker
  export_root=$(realpath ../sysroot_docker)
  mkdir -p "${export_root}/usr_s100"
  export_index=$(mktemp /tmp/s100-sysroot-index.XXXXXX)
  rm -f "${export_index}"
  GIT_INDEX_FILE="${export_index}" \
    git -C "${cache}" read-tree "${ref}:usr_s100"
  GIT_INDEX_FILE="${export_index}" \
    git -C "${cache}" checkout-index \
      --all --force --prefix="${export_root}/usr_s100/"
  rm -f "${export_index}"
  test -d ../sysroot_docker/usr_s100
  printf '%s\n' "${ref}" > ../sysroot_docker/.s100-sysroot-ref
}

echo "=== Fetch required ROS/TROS dependencies ==="
checkout_sysroot_in_batches "${S100_SYSROOT_REF}"
shallow_checkout \
  "${TROS_ARM_BUILD_URL}" \
  develop \
  tros_arm_build

echo "=== Verify pinned S100 sysroot ==="
test -d ../sysroot_docker/usr_s100
actual_sysroot_ref=$(<../sysroot_docker/.s100-sysroot-ref)
test "${actual_sysroot_ref}" = "${S100_SYSROOT_REF}"
echo "sysroot revision: ${actual_sysroot_ref}"

echo "=== Verify official TROS build environment ==="
for required_command in \
  aarch64-linux-gnu-gcc \
  aarch64-linux-gnu-g++ \
  cmake \
  colcon \
  curl \
  file \
  readelf; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "ERROR: official TROS image is missing: ${required_command}" >&2
    exit 1
  fi
done
aarch64-linux-gnu-gcc --version
test -f /opt/ros/humble/setup.bash

echo "=== Complete ARM64 Qt5 runtime in target sysroot ==="
qt_core=../sysroot_docker/usr_s100/lib/aarch64-linux-gnu/libQt5Core.so.5.15.3

if [[ ! -e "${qt_core}" ]]; then
  qt_deb_dir=/mnt/s100-cross/cache/qt-arm64
  qt_extract_root=/tmp/s100-qt-root
  arm64_apt_dir=/tmp/s100-arm64-apt
  mkdir -p "${qt_deb_dir}" "${qt_extract_root}" "${arm64_apt_dir}/lists/partial"

  printf '%s\n' \
    'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main universe restricted multiverse' \
    'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe restricted multiverse' \
    'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-security main universe restricted multiverse' \
    > "${arm64_apt_dir}/sources.list"

  arm64_apt_options=(
    -o "APT::Architecture=arm64"
    -o "APT::Architectures=arm64"
    -o "Dir::Etc::sourcelist=${arm64_apt_dir}/sources.list"
    -o "Dir::Etc::sourceparts=-"
    -o "Dir::State::lists=${arm64_apt_dir}/lists"
  )

  if ! compgen -G "${qt_deb_dir}/python3-vtk9_*_arm64.deb" >/dev/null; then
    apt-get "${arm64_apt_options[@]}" update
    cd "${qt_deb_dir}"
    apt-get "${arm64_apt_options[@]}" download \
      qtbase5-dev:arm64 \
      libqt5core5a:arm64 \
      libqt5dbus5:arm64 \
      libqt5network5:arm64 \
      libqt5gui5:arm64 \
      libqt5widgets5:arm64 \
      libqt5opengl5:arm64 \
      libqt5opengl5-dev:arm64 \
      libqt5printsupport5:arm64 \
      libqt5concurrent5:arm64 \
      libqt5test5:arm64 \
      qt5-qmake:arm64 \
      qt5-qmake-bin:arm64 \
      qtbase5-dev-tools:arm64 \
      python3-vtk9:arm64
  fi

  # python3-vtk9 does not make `apt-get download` fetch its dependencies.
  # PCL's libpcl_io has direct DT_NEEDED entries for the libraries shipped by
  # libvtk9.1, so cache that runtime package explicitly as well.
  if ! compgen -G "${qt_deb_dir}/libvtk9.1_*_arm64.deb" >/dev/null; then
    apt-get "${arm64_apt_options[@]}" update
    cd "${qt_deb_dir}"
    apt-get "${arm64_apt_options[@]}" download libvtk9.1:arm64
  fi

  while IFS= read -r -d '' deb; do
    dpkg-deb --extract "${deb}" "${qt_extract_root}"
  done < <(find "${qt_deb_dir}" -maxdepth 1 -name '*.deb' -print0)

  # Debian packages are rooted at /usr. The D-Robotics sysroot already treats
  # usr_s100 as its /usr, so merge the contents rather than adding usr_s100/usr.
  cp -a "${qt_extract_root}/usr/." \
    /mnt/s100-cross/cc_ws/sysroot_docker/usr_s100/

  cd /mnt/s100-cross/cc_ws/tros_ws
fi

for qt_library in \
  libQt5Core.so.5.15.3 \
  libQt5Gui.so.5.15.3 \
  libQt5Widgets.so.5.15.3 \
  libQt5OpenGL.so.5.15.3; do
  qt_path="../sysroot_docker/usr_s100/lib/aarch64-linux-gnu/${qt_library}"
  if [[ ! -e "${qt_path}" ]]; then
    echo "ERROR: ARM64 Qt library was not installed in the sysroot: ${qt_path}" >&2
    exit 1
  fi
  if ! file "${qt_path}" | grep -Eq 'ARM aarch64|ARM64'; then
    echo "ERROR: Qt library is not an ARM64 binary: ${qt_path}" >&2
    file "${qt_path}" >&2
    exit 1
  fi
done
test -x ../sysroot_docker/usr_s100/lib/aarch64-linux-gnu/qt5/bin/qmake
test -x ../sysroot_docker/usr_s100/lib/qt5/bin/moc
test -x ../sysroot_docker/usr_s100/bin/pvtkpython
test -f ../sysroot_docker/usr_s100/lib/aarch64-linux-gnu/libvtkCommonCore-9.1.so.1

echo "=== Complete ARM64 ROS PCL packages in target TROS ==="
pcl_ros_config=/opt/ros/humble/share/pcl_ros/cmake/pcl_rosConfig.cmake
if [[ ! -f "${pcl_ros_config}" ]]; then
  ros_deb_dir=/mnt/s100-cross/cache/ros-pcl-arm64
  ros_extract_root=/tmp/s100-ros-pcl-root
  ros_arm64_apt_dir=/tmp/s100-ros-arm64-apt
  mkdir -p \
    "${ros_deb_dir}" \
    "${ros_extract_root}" \
    "${ros_arm64_apt_dir}/lists/partial"

  if ! compgen -G "${ros_deb_dir}/ros-humble-pcl-ros_*_arm64.deb" >/dev/null; then
    curl --fail --location \
      https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      --output "${ros_arm64_apt_dir}/ros.key"
    printf '%s\n' \
      "deb [arch=arm64 signed-by=${ros_arm64_apt_dir}/ros.key] http://packages.ros.org/ros2/ubuntu jammy main" \
      > "${ros_arm64_apt_dir}/sources.list"

    ros_arm64_apt_options=(
      -o "APT::Architecture=arm64"
      -o "APT::Architectures=arm64"
      -o "Dir::Etc::sourcelist=${ros_arm64_apt_dir}/sources.list"
      -o "Dir::Etc::sourceparts=-"
      -o "Dir::State::lists=${ros_arm64_apt_dir}/lists"
    )

    apt-get "${ros_arm64_apt_options[@]}" update
    cd "${ros_deb_dir}"
    apt-get "${ros_arm64_apt_options[@]}" download \
      ros-humble-pcl-conversions:arm64 \
      ros-humble-pcl-ros:arm64
  fi

  while IFS= read -r -d '' deb; do
    dpkg-deb --extract "${deb}" "${ros_extract_root}"
  done < <(find "${ros_deb_dir}" -maxdepth 1 -name '*.deb' -print0)

  cp -a "${ros_extract_root}/opt/ros/humble/." /opt/ros/humble/
  cd /mnt/s100-cross/cc_ws/tros_ws
fi

test -f "${pcl_ros_config}"
test -f /opt/ros/humble/share/pcl_conversions/cmake/pcl_conversionsConfig.cmake
test -f /opt/ros/humble/lib/libpcl_ros_tf.a
file /opt/ros/humble/lib/libpcd_to_pointcloud_lib.so \
  | grep -Eq 'ARM aarch64|ARM64'

echo "=== Cross-compile GTSAM into the S100 system directory ==="
# fast_lio's loop-closure backend needs GTSAM >= 4.2, which the D-Robotics
# sysroot does not ship. Build the same release as ros-humble-gtsam (x86) with
# prefix /usr into usr_s100, the board's /usr. Boost 1.74, Eigen 3 and TBB
# already come from the pinned sysroot.
gtsam_version=4.2.0
gtsam_sha256=8b44d6b98a3b608664d1c9a7c1383a406550499d894533bb0183e6cf487e6457
gtsam_config=../sysroot_docker/usr_s100/lib/cmake/GTSAM/GTSAMConfig.cmake
if [[ ! -f "${gtsam_config}" ]]; then
  gtsam_cache=/mnt/s100-cross/cache/gtsam
  gtsam_archive=${gtsam_cache}/gtsam-${gtsam_version}.tar.gz
  gtsam_work=/tmp/s100-gtsam
  mkdir -p "${gtsam_cache}"
  if [[ ! -f "${gtsam_archive}" ]]; then
    curl --fail --location --retry 5 --retry-all-errors \
      --output "${gtsam_archive}.partial" \
      "https://github.com/borglab/gtsam/archive/refs/tags/${gtsam_version}.tar.gz"
    mv "${gtsam_archive}.partial" "${gtsam_archive}"
  fi
  echo "${gtsam_sha256}  ${gtsam_archive}" | sha256sum --check
  rm -rf "${gtsam_work}"
  mkdir -p "${gtsam_work}"
  tar -xzf "${gtsam_archive}" -C "${gtsam_work}"

  # The toolchain file resolves the sysroot through sysroot_docker/usr, which
  # robot_dev_config/build.sh normally creates (and removes first) itself.
  ln -sfn "$(realpath ../sysroot_docker)/usr_s100" ../sysroot_docker/usr

  CROSS_COMPILE=/usr/bin/aarch64-linux-gnu- cmake \
    -S "${gtsam_work}/gtsam-${gtsam_version}" \
    -B "${gtsam_work}/build" \
    -DCMAKE_TOOLCHAIN_FILE="$(pwd)/robot_dev_config/aarch64_toolchainfile.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_WITH_TBB=ON \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_UNSTABLE=OFF \
    -DGTSAM_BUILD_PYTHON=OFF \
    -DGTSAM_INSTALL_CPPUNITLITE=OFF
  CROSS_COMPILE=/usr/bin/aarch64-linux-gnu- \
    cmake --build "${gtsam_work}/build" --parallel "$(nproc)"
  DESTDIR="${gtsam_work}/stage" cmake --install "${gtsam_work}/build"
  cp -a "${gtsam_work}/stage/usr/." ../sysroot_docker/usr_s100/
  rm -rf "${gtsam_work}"
fi
test -f "${gtsam_config}"
grep -q "${gtsam_version}" ../sysroot_docker/usr_s100/lib/cmake/GTSAM/GTSAMConfigVersion.cmake
gtsam_library=$(find -L ../sysroot_docker/usr_s100/lib -maxdepth 1 \
  -name 'libgtsam.so.*' -type f -print -quit)
test -n "${gtsam_library}"
file "${gtsam_library}" | grep -Eq 'ARM aarch64|ARM64'

echo "=== Activate ROS 2 build environment ==="
set +u
source /opt/ros/humble/setup.bash
set -u

echo "=== Build S100 ==="
bash robot_dev_config/build.sh \
  -p S100 \
  -c '-DUSE_LIVOX_CUSTOM_MSG=OFF' \
  -r '--packages-up-to fast_lio icp_relocalization omni_slam_interfaces omni_slam_manager omni_tf_manager'

# robot_dev_config/build.sh swallows the colcon exit status. These checks make
# the shared script fail if the real target outputs were not produced.
test -x install/lib/fast_lio/fastlio_mapping
test -x install/lib/icp_relocalization/icp_node
test -f install/share/omni_slam_interfaces/package.xml
test -f install/share/omni_slam_manager/package.xml
test -f install/share/omni_robot_interfaces/package.xml
test -x install/lib/omni_tf_manager/omni_tf_manager_node
test -f install/share/omni_tf_manager/config/generic_four_sensor.yaml
test -f install/share/omni_tf_manager/config/omni_dog.yaml
test -f install/share/omni_tf_manager/config/omni_vbot_dog.yaml
bash src/omni_slam/scripts/validate_vbot_tf_profile.sh \
  install/share/omni_tf_manager/config/omni_vbot_dog.yaml

echo "=== Verify output architecture ==="
elf_count=0
while IFS= read -r file_path; do
  file_info=$(file "${file_path}")
  if grep -q ELF <<<"${file_info}"; then
    echo "${file_info}"
    grep -Eq 'ARM aarch64|ARM64' <<<"${file_info}"
    elf_count=$((elf_count + 1))
  fi
done < <(find install -type f -perm /111)

if (( elf_count == 0 )); then
  echo 'ERROR: no executable ARM64 ELF files were produced' >&2
  exit 1
fi

echo "=== Assemble deployable S100 runtime ==="
bash src/omni_slam/scripts/package_s100_runtime.sh \
  install \
  ../sysroot_docker \
  runtime

test -x runtime/bin/fastlio_mapping
test -x runtime/bin/icp_node
test -d runtime/config/fast_lio
test -s runtime/lib/DEPENDENCIES.txt
bash src/omni_slam/scripts/validate_vbot_tf_profile.sh \
  runtime/config/omni_tf_manager/omni_vbot_dog.yaml
test -f runtime/local/lib/python3.10/dist-packages/omni_slam_interfaces/__init__.py
find runtime/local/lib/python3.10/dist-packages/omni_slam_interfaces \
  -maxdepth 1 -type f -name '*rosidl_typesupport_c*.so' -print -quit \
  | grep -q .

du -sh install
echo "S100 cross build completed"
CONTAINER

log "build output: ${TROS_ROOT}/install"
log "runtime bundle: ${TROS_ROOT}/runtime"
