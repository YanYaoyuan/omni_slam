# omni_slam_manager

This package is the engineering control plane for FAST-LIO and publishes the
typed state consumed by `omni_tf_manager`. `MapStore` is its immutable,
crash-safe map asset store.

Layout:

```text
<map_root>/<map_id>/
├── current -> versions/v000002
└── versions/
    ├── v000001/
    │   ├── map.pcd
    │   └── manifest.json
    └── v000002/
        ├── map.pcd
        └── manifest.json
```

An import copies and hashes the PCD into a staging directory, fsyncs the PCD,
manifest and directory, atomically publishes the immutable version directory,
then atomically switches `current`.  Failed imports never switch the active
map.  `MapStore.current()` verifies SHA-256 by default before returning a map.

The ROS state machine, typed status and actions will be layered on this storage
primitive; callers should not modify files inside `versions/`.

Start the manager once and leave it running. For Vbot on S100:

```bash
ros2 launch omni_slam_manager manager.launch.py \
  robot_profile:=vbot \
  map_root:=/userdata/omni/maps \
  map_source_path:=/userdata/omni/work/map.pcd
```

The manager selects `vbot.launch.py`, `vbot_relocalization.launch.py` and
the matching `omni_vbot_dog.yaml` TF profile. The default `omni_dog` robot
selects `omni_dog.yaml`. An explicit `tf_profile_path` still overrides this
selection.

`tf_profile_path` is the single source of truth for map, odom, base and IMU
tracking frame names. The launch also starts `omni_tf_manager`; the SLAM child
processes only publish pose data and never publish managed TF edges.
FAST-LIO consumes only `/omni/sensors/lidar/points` and
`/omni/sensors/imu/data`; the TF profile owns each vendor-topic alias.

Stable control-plane endpoints:

- `/omni/slam/start_mapping` (`StartMapping`)
- `/omni/slam/stop_mapping` (`StopMapping`)
- `/omni/slam/save_map` (`SaveMap` action)
- `/omni/slam/start_localization` (`StartLocalization` action)
- `/omni/slam/stop_localization` (`StopLocalization`)
- `/omni/slam/status` (`omni_tf_manager/msg/SlamStatus`, reliable
  transient-local heartbeat)

The manager starts child launch processes without a shell and terminates their
entire process group on stop, cancellation or shutdown.  Mapping is not marked
ready until cloud, IMU and odometry telemetry are all fresh.  Localization is
not marked `LOCALIZED` until an ICP initial pose has been followed by fresh
odometry.

## Vbot control flow

The S100 runtime installs `omni_slam_ctl`, which wraps all calls below and
keeps the manager in the background. For normal board operation, prefer:

```bash
omni_slam_ctl manager-start
omni_slam_ctl mapping-start vbot_001
omni_slam_ctl map-save factory_a
omni_slam_ctl mapping-stop
omni_slam_ctl localization-start factory_a
omni_slam_ctl logs-follow
```

Run `omni_slam_ctl help` for stop, status, initial-pose and log commands. The
raw ROS 2 commands remain documented below for integration with other nodes.

Start mapping:

```bash
ros2 service call /omni/slam/start_mapping \
  omni_slam_interfaces/srv/StartMapping \
  "{session_id: vbot_001}"
```

Save the current map as a verified, versioned asset named `factory_a`:

```bash
ros2 action send_goal --feedback /omni/slam/save_map \
  omni_slam_interfaces/action/SaveMap \
  "{map_id: factory_a, calibration_hash: '', make_current: true}"
```

Stop mapping after the save succeeds:

```bash
ros2 service call /omni/slam/stop_mapping \
  omni_slam_interfaces/srv/StopMapping \
  "{discard_unsaved: false}"
```

Start localization with the current version of `factory_a`. Initial position
is in meters and yaw is in radians:

```bash
ros2 action send_goal --feedback /omni/slam/start_localization \
  omni_slam_interfaces/action/StartLocalization \
  "{map_id: factory_a, map_version: 0, timeout_sec: 60.0, initial_x: 0.0, initial_y: 0.0, initial_z: 0.0, initial_yaw: 0.0}"
```

Stop localization:

```bash
ros2 service call /omni/slam/stop_localization \
  omni_slam_interfaces/srv/StopLocalization "{}"
```

Observe the durable status heartbeat at any time:

```bash
ros2 topic echo /omni/slam/status
```
