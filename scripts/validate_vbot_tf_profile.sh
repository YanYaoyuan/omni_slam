#!/usr/bin/env bash

# Fail fast when an omni_tf_manager revision would silently suppress the
# canonical Vbot sensor topics consumed by FAST-LIO.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <omni_vbot_dog.yaml>" >&2
  exit 2
fi

profile=$1
if [[ ! -f "${profile}" ]]; then
  echo "ERROR: Vbot TF profile not found: ${profile}" >&2
  exit 1
fi

require_setting() {
  local key=$1
  local expected=$2
  local setting="    ${key}: ${expected}"
  local count
  count=$(grep -Fxc "${setting}" "${profile}" || true)
  if [[ "${count}" != "1" ]]; then
    echo "ERROR: expected one Vbot TF setting '${setting}' in ${profile}" >&2
    exit 1
  fi
}

require_setting mode shadow
require_setting profile_name omni_vbot_dog
require_setting sensor_relay.lidar.operation identity_frame_alias
require_setting sensor_relay.lidar.alias_verified true
require_setting sensor_relay.lidar.input_topic /lidar_points
require_setting sensor_relay.lidar.output_topic /omni/sensors/lidar/points
require_setting sensor_relay.lidar.input_frame vita_lidar
require_setting sensor_relay.lidar.output_frame omni_lidar_link
require_setting sensor_relay.imu.operation identity_frame_alias
require_setting sensor_relay.imu.alias_verified true
require_setting sensor_relay.imu.input_topic /lidar_imu
require_setting sensor_relay.imu.output_topic /omni/sensors/imu/data
require_setting sensor_relay.imu.input_frame vita_lidar
require_setting sensor_relay.imu.output_frame omni_imu_link

if grep -Fq 'PENDING_LIVE_HEADER_VERIFICATION' "${profile}"; then
  echo "ERROR: Vbot TF profile still carries a pending live-header marker" >&2
  exit 1
fi

echo "Vbot TF profile publishes reviewed vita_lidar canonical aliases"
