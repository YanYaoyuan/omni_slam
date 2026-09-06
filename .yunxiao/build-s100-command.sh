#!/usr/bin/env bash

set -euo pipefail

# Install tools only when the build image does not already provide them.
missing_tools=0
for tool in curl docker git rsync sha256sum tar; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing_tools=1
  fi
done
if [ "$missing_tools" -ne 0 ]; then
  dnf install -y curl docker git rsync tar gzip coreutils findutils
fi

docker info

S100_IMAGE="pc_tros_ubuntu22.04:v1.0.0"
S100_IMAGE_URL="http://archive.d-robotics.cc/TogetheROS/cross_compile_docker/pc_tros_ubuntu22.04_v1.0.0.tar.gz"
S100_IMAGE_SHA256="ec065c268782373311421f0bb7f2212bfec6584a7070a9737468055a6bce2c88"
ROBOT_DEV_CONFIG_REF="f44b1ad2575b34c189fd470de1db9c5ad48b6886"
S100_SYSROOT_REF="de9fa286f71a72d24c349477cd59f41cc2cc3d8f"
TF_MANAGER_REF="${TF_MANAGER_REF:-main}"
VERSION_INPUT="${VERSION_INPUT:-}"

source_root=$(git rev-parse --show-toplevel)
cd "$source_root"

test -f "$source_root/scripts/validate_vbot_tf_profile.sh"
test -f "$source_root/scripts/build_s100_cross.sh"
test -f "$source_root/.github/ci/write_manifest.sh"
echo "Omni SLAM source root: $source_root"

# The Beijing public runner has unstable access to GitHub. Pin all three
# D-Robotics repositories directly to their internal Codeup mirrors.
sed -i \
  -e 's#https://github.com/D-Robotics/sysroot_docker.git#https://codeup.aliyun.com/bgyfw/AI-Platform/robotics/sysroot_docker.git#g' \
  -e 's#https://github.com/D-Robotics/robot_dev_config.git#https://codeup.aliyun.com/bgyfw/AI-Platform/robotics/robot_dev_config.git#g' \
  -e 's#https://github.com/D-Robotics/tros_arm_build.git#https://codeup.aliyun.com/bgyfw/AI-Platform/robotics/tros_arm_build.git#g' \
  -e 's#http://packages.ros.org/ros2/ubuntu#https://mirrors.ustc.edu.cn/ros2/ubuntu#g' \
  "$source_root/scripts/build_s100_cross.sh"

# Keep this runtime patch until the native credential-file support in
# build_s100_cross.sh has been committed to the branch used by Flow.
if ! grep -q '/mnt/s100-cross/.codeup-git-credentials' \
    "$source_root/scripts/build_s100_cross.sh"; then
  sed -i \
    '/export DEBIAN_FRONTEND=noninteractive/a git config --global credential.helper "store --file=/mnt/s100-cross/.codeup-git-credentials"' \
    "$source_root/scripts/build_s100_cross.sh"
fi

# Keep the work tree beside the source checkout. CloudEffect shares its
# workspace with the Docker daemon, while /tmp may belong only to the command
# container and therefore cannot reliably be bind-mounted by docker run.
build_number="${BUILD_NUMBER:-manual}"
work_root="$(dirname "$source_root")/.omni-slam-s100-$build_number"
tf_root="$work_root/omni_tf_manager_source"
image_archive="$work_root/pc_tros_ubuntu22.04_v1.0.0.tar.gz"

mkdir -p "$work_root"
rm -rf "$tf_root"

# CODEUP_USERNAME and CODEUP_TOKEN must be configured as private Flow
# variables. The temporary credential is visible only to this build and the
# nested TROS container, and is removed when the command step exits.
: "${CODEUP_USERNAME:?configure the private Flow variable CODEUP_USERNAME}"
: "${CODEUP_TOKEN:?configure the private Flow variable CODEUP_TOKEN}"

credential_file="$work_root/s100-cross/.codeup-git-credentials"
mkdir -p "$(dirname "$credential_file")"
umask 077
printf 'protocol=https\nhost=codeup.aliyun.com\nusername=%s\npassword=%s\n\n' \
  "$CODEUP_USERNAME" "$CODEUP_TOKEN" \
  | git -c credential.helper="store --file=$credential_file" \
      credential approve

cleanup_codeup_credentials() {
  rm -f "$credential_file"
}
trap cleanup_codeup_credentials EXIT

git clone \
  --filter=blob:none \
  --no-checkout \
  https://codeup.aliyun.com/bgyfw/AI-Platform/robotics/omni_tf_manager.git \
  "$tf_root"

git -C "$tf_root" fetch \
  --no-tags \
  --depth=1 \
  origin \
  "$TF_MANAGER_REF"
git -C "$tf_root" checkout --detach FETCH_HEAD

tf_manager_sha=$(git -C "$tf_root" rev-parse HEAD)
echo "Omni TF Manager revision: $tf_manager_sha"

bash "$source_root/scripts/validate_vbot_tf_profile.sh" \
  "$tf_root/config/omni_vbot_dog.yaml"

curl \
  --fail \
  --location \
  --retry 5 \
  --continue-at - \
  --output "$image_archive" \
  "$S100_IMAGE_URL"

echo "$S100_IMAGE_SHA256  $image_archive" | sha256sum --check

docker load --input "$image_archive"
docker image inspect "$S100_IMAGE" \
  --format='ID={{.Id}} ARCH={{.Architecture}} OS={{.Os}}'

export S100_SOURCE_ROOT="$source_root"
export OMNI_TF_MANAGER_SOURCE="$tf_root"
export S100_WORK_ROOT="$work_root/s100-cross"
export S100_CLEAN=1
export ROBOT_DEV_CONFIG_REF
export S100_SYSROOT_REF
export S100_IMAGE

bash "$source_root/scripts/build_s100_cross.sh"

runtime_root="$S100_WORK_ROOT/cc_ws/tros_ws/runtime"

test -x "$runtime_root/bin/fastlio_mapping"
test -x "$runtime_root/bin/icp_node"
test -x "$runtime_root/bin/omni_tf_manager_node"
test -f "$runtime_root/config/omni_tf_manager/generic_four_sensor.yaml"
test -f "$runtime_root/config/omni_tf_manager/omni_dog.yaml"
test -f "$runtime_root/config/omni_tf_manager/omni_vbot_dog.yaml"

bash "$source_root/scripts/validate_vbot_tf_profile.sh" \
  "$runtime_root/config/omni_tf_manager/omni_vbot_dog.yaml"

test -d "$runtime_root/config/fast_lio"
test -s "$runtime_root/lib/DEPENDENCIES.txt"

python_package="$runtime_root/local/lib/python3.10/dist-packages/omni_slam_interfaces"
test -f "$python_package/__init__.py"
find "$python_package" \
  -maxdepth 1 \
  -type f \
  -name '*rosidl_typesupport_c*.so' \
  -print -quit | grep -q .

commit_sha=$(git -C "$source_root" rev-parse HEAD)
short_sha=$(printf '%s' "$commit_sha" | cut -c1-12)

export GITHUB_SHA="$commit_sha"
export OMNI_TF_MANAGER_SHA="$tf_manager_sha"

version="$VERSION_INPUT"
if [ -z "$version" ]; then
  version="$short_sha"
fi
version=$(printf '%s' "$version" | tr '/' '-')

artifact_name="omni-slam-rdk-s100-aarch64-$version-$short_sha"

rm -rf "$source_root/dist"
mkdir -p "$source_root/dist/$artifact_name"
cp -a "$runtime_root/." "$source_root/dist/$artifact_name/"

bash "$source_root/.github/ci/write_manifest.sh" \
  aarch64 \
  rdk-s100-rdkos-4 \
  "$version" \
  "$source_root/dist/$artifact_name"

tar \
  -C "$source_root/dist" \
  -czf "$source_root/dist/$artifact_name.tar.gz" \
  "$artifact_name"

ls -lh "$source_root/dist/$artifact_name.tar.gz"
