#!/usr/bin/env bash

# Assemble a deployable S100 application prefix from a colcon install tree.
# Run this inside the official cross-compilation container so target ROS
# libraries under /opt/ros/humble are available to the dependency resolver.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <install-root> <sysroot> <output-root>" >&2
  exit 2
fi

install_root=$(realpath "$1")
sysroot=$(realpath "$2")
output_root=$(realpath -m "$3")

if [[ ! -d "${install_root}/lib" || ! -d "${install_root}/share" ]]; then
  echo "ERROR: invalid colcon install tree: ${install_root}" >&2
  exit 1
fi
if [[ ! -d "${sysroot}/usr" ]]; then
  echo "ERROR: invalid S100 sysroot: ${sysroot}" >&2
  exit 1
fi
case "${output_root}" in
  /|"${install_root}"|"${sysroot}")
    echo "ERROR: unsafe runtime output path: ${output_root}" >&2
    exit 1
    ;;
esac

rm -rf "${output_root}"
mkdir -p \
  "${output_root}/bin" \
  "${output_root}/config" \
  "${output_root}/lib" \
  "${output_root}/share"

# Preserve the ROS prefix layout used by ament and ros2 launch. Application
# executables remain below lib/<package>; bin contains convenient symlinks.
cp -a "${install_root}/lib/." "${output_root}/lib/"
cp -a "${install_root}/share/." "${output_root}/share/"

for package_name in fast_lio icp_relocalization omni_slam_interfaces omni_slam_manager omni_tf_manager; do
  package_lib="${output_root}/lib/${package_name}"
  if [[ -d "${package_lib}" ]]; then
    while IFS= read -r -d '' executable; do
      executable_name=$(basename "${executable}")
      ln -sfn "../lib/${package_name}/${executable_name}" \
        "${output_root}/bin/${executable_name}"
    done < <(find "${package_lib}" -maxdepth 1 -type f -perm /111 -print0)
  fi

  package_config="${install_root}/share/${package_name}/config"
  if [[ -d "${package_config}" ]]; then
    mkdir -p "${output_root}/config/${package_name}"
    cp -a "${package_config}/." "${output_root}/config/${package_name}/"
  fi
done

cat > "${output_root}/setup.bash" <<'SETUP'
#!/usr/bin/env bash

_omni_s100_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Use the TROS installation already present on the S100. The archive carries
# application-specific shared libraries, but deliberately does not replace
# the board's glibc or ROS command-line tools.
if [[ -f /opt/tros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/tros/humble/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

export PATH="${_omni_s100_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${_omni_s100_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export AMENT_PREFIX_PATH="${_omni_s100_root}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"

if [[ -d "${_omni_s100_root}/lib/python3.10/site-packages" ]]; then
  export PYTHONPATH="${_omni_s100_root}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
fi

unset _omni_s100_root
SETUP
chmod 755 "${output_root}/setup.bash"

cat > "${output_root}/README.txt" <<'README'
Omni SLAM for RDK S100

1. Copy this directory to the S100.
2. Run: source setup.bash
3. Run: omni_slam_ctl manager-start
4. Run: omni_slam_ctl help

Logs from the manager and its mapping/localization child processes are under
/userdata/omni/log. Use `omni_slam_ctl logs-follow` to follow the current log.

The bundle includes application-specific ARM64 shared-library dependencies.
The board must provide RDK OS and ROS 2 Humble/TogetheROS under /opt/tros/humble
or /opt/ros/humble.
README

# Libraries supplied by the target operating system must not be placed ahead
# of the board's own glibc in LD_LIBRARY_PATH.
is_system_library() {
  case "$1" in
    ld-linux-aarch64.so.1|libanl.so.*|libc.so.*|libdl.so.*|libm.so.*|\
      libnsl.so.*|libpthread.so.*|libresolv.so.*|librt.so.*|libutil.so.*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

search_roots=(
  /opt/ros/humble/lib
  /opt/ros/humble/lib/aarch64-linux-gnu
  "${sysroot}/usr/lib/aarch64-linux-gnu"
  "${sysroot}/usr/lib"
  "${sysroot}/lib/aarch64-linux-gnu"
  /usr/aarch64-linux-gnu/lib
)

resolve_library() {
  local library_name=$1
  local search_root
  local candidate

  for search_root in "${search_roots[@]}"; do
    [[ -d "${search_root}" ]] || continue
    candidate=$(find -L "${search_root}" -maxdepth 2 -type f \
      -name "${library_name}" -print -quit 2>/dev/null || true)
    if [[ -n "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

declare -a elf_queue=()
declare -A queued=()
declare -A processed=()

while IFS= read -r -d '' candidate; do
  if file "${candidate}" | grep -q 'ELF'; then
    elf_queue+=("${candidate}")
    queued["${candidate}"]=1
  fi
done < <(find "${output_root}/lib" -type f -print0)

dependency_manifest="${output_root}/lib/DEPENDENCIES.txt"
: > "${dependency_manifest}"

queue_index=0
while (( queue_index < ${#elf_queue[@]} )); do
  elf_file=${elf_queue[queue_index]}
  queue_index=$((queue_index + 1))

  [[ -z "${processed[${elf_file}]:-}" ]] || continue
  processed["${elf_file}"]=1

  while IFS= read -r library_name; do
    [[ -n "${library_name}" ]] || continue
    if is_system_library "${library_name}"; then
      continue
    fi

    bundled_library="${output_root}/lib/${library_name}"
    if [[ ! -f "${bundled_library}" ]]; then
      if ! source_library=$(resolve_library "${library_name}"); then
        echo "ERROR: cannot resolve ARM64 runtime dependency ${library_name}" \
          "required by ${elf_file}" >&2
        exit 1
      fi
      cp -L "${source_library}" "${bundled_library}"
      chmod 755 "${bundled_library}"
      printf '%s <- %s\n' "${library_name}" "${source_library}" \
        >> "${dependency_manifest}"
    fi

    if [[ -z "${queued[${bundled_library}]:-}" ]]; then
      elf_queue+=("${bundled_library}")
      queued["${bundled_library}"]=1
    fi
  done < <(
    readelf -d "${elf_file}" 2>/dev/null \
      | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p'
  )
done

sort -u -o "${dependency_manifest}" "${dependency_manifest}"

test -x "${output_root}/bin/fastlio_mapping"
test -x "${output_root}/bin/icp_node"
test -x "${output_root}/bin/omni_slam_ctl"
test -x "${output_root}/bin/omni_tf_manager_node"
test -d "${output_root}/config/fast_lio"
test -f "${output_root}/config/omni_tf_manager/generic_four_sensor.yaml"
test -f "${output_root}/config/omni_tf_manager/omni_dog.yaml"
test -f "${output_root}/config/omni_tf_manager/omni_vbot_dog.yaml"
test -d "${output_root}/share/ament_index"

elf_count=0
while IFS= read -r -d '' bundled_file; do
  file_info=$(file "${bundled_file}")
  if grep -q 'ELF' <<<"${file_info}"; then
    if ! grep -Eq 'ARM aarch64|ARM64' <<<"${file_info}"; then
      echo "ERROR: non-ARM64 ELF in runtime bundle: ${file_info}" >&2
      exit 1
    fi
    elf_count=$((elf_count + 1))
  fi
done < <(find "${output_root}" -type f -print0)

if (( elf_count == 0 )); then
  echo 'ERROR: runtime bundle contains no ARM64 ELF files' >&2
  exit 1
fi

echo "Runtime bundle: ${output_root}"
echo "ARM64 ELF files: ${elf_count}"
du -sh "${output_root}"
