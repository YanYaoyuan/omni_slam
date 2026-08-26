#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
rm -rf build install log
colcon build --merge-install \
  --packages-select fast_lio icp_relocalization omni_slam_interfaces omni_slam_manager omni_tf_manager \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DUSE_LIVOX_CUSTOM_MSG=OFF
source install/setup.bash
colcon test --merge-install \
  --packages-select fast_lio icp_relocalization omni_slam_manager omni_tf_manager \
  --event-handlers console_direct+
colcon test-result --verbose
test -f install/share/omni_tf_manager/config/omni_dog.yaml
test -f install/share/omni_tf_manager/config/omni_vbot_dog.yaml
scripts/validate_vbot_tf_profile.sh \
  install/share/omni_tf_manager/config/omni_vbot_dog.yaml
test -f install/share/fast_lio/config/omni_dog.yaml
test -f install/share/fast_lio/config/vbot.yaml
