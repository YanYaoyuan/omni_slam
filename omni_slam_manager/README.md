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
