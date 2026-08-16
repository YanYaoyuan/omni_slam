# omni_slam_manager

This package is the engineering control plane for FAST-LIO.  The first
implemented component is `MapStore`, an immutable, crash-safe map asset store.

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

Start the manager without starting FAST-LIO:

```bash
ros2 launch omni_slam_manager manager.launch.py \
  map_root:=/var/lib/omni/maps \
  map_source_path:=/tmp/omni_slam_work/map.pcd
```

Stable control-plane endpoints:

- `/omni/slam/start_mapping` (`StartMapping`)
- `/omni/slam/stop_mapping` (`StopMapping`)
- `/omni/slam/save_map` (`SaveMap` action)
- `/omni/slam/start_localization` (`StartLocalization` action)
- `/omni/slam/status` (`SlamStatus`, reliable transient-local heartbeat)

The manager starts child launch processes without a shell and terminates their
entire process group on stop, cancellation or shutdown.  Mapping is not marked
ready until cloud, IMU and odometry telemetry are all fresh.  Localization is
not marked `LOCALIZED` until an ICP initial pose has been followed by fresh
odometry.
