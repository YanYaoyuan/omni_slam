# Upstream source provenance

Cloned on 2026-11-05 for a native ROS 2 Humble port.

- FAST-LIVO2: https://github.com/hku-mars/FAST-LIVO2.git, commit `0d2c0346107b75b59934975adec9a6eeeb913c64`.
- Sophus: https://github.com/strasdat/Sophus.git, commit `a621ff2e56c56c839a6c40418d42c3c254424b5c`.
- Vikit: https://github.com/xuankuzcr/rpg_vikit.git, commit `6c886c8e5d83997806e00294826d528cea3581dd`.

Upstream notices and license files are retained. Sophus SO2 complex setters are adapted for the current C++ standard. Vikit is built as a private camera/math library, without its ROS 1 loader. The estimator is based on the original implementation; transport, parameters, timestamps, runtime paths, VBot preprocessing and experiment observability are adapted locally. Retained LiDAR points are additionally split and rebased at each image time rather than consumed before their acquisition time; this local synchronization fix is covered by scan-window regression tests.

The subsequent VBot adaptation also initializes IMU integration time explicitly, consumes initialization windows consistently, prevents repeated deskew of the first point, fixes reference-patch NCC statistics and discarded-patch ownership, validates photometric sampling bounds at each pyramid/search level, and sums parallel residual errors in a stable order. An opt-in `vio.lidar_timed_fusion` schedule keeps complete LiDAR scans and attaches a recent camera measurement at the LiDAR epoch using IMU rotation and short-interval constant-velocity translation compensation. This schedule and the conservative photometric parameters are local experimental changes, not claims about the upstream method's benchmark accuracy.

The vertical-consistency follow-up adds an explicit IMU acceleration unit scale (0 retains upstream automatic normalization; 1 preserves SI input), native IMU-frame XYZRGB keyframe archives, and a separate offline Open3D pose-graph experiment. The latter estimates a constant gravity-frame transform, validates 3D submap revisits, reserves some revisit measurements from graph fitting, and exports corrected keyframe maps without feeding corrections into the frontend. It does not impose a flat-ground or zero-height constraint, and internal revisit consistency is not trajectory ground truth. These are local additions rather than upstream ROS 2 or loop-closure capabilities.
