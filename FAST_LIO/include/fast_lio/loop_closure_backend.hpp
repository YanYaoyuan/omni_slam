#pragma once

#include "fast_lio/map_archive.hpp"
#include <memory>

namespace fast_lio {

// Owns the mapping-session archive and a background pose graph. Corrections
// affect exported maps/preview only; no callback can mutate FAST-LIO state.
class LoopClosureBackend {
public:
  using Pose = MapArchive::Pose;
  using Cloud = MapArchive::Cloud;
  struct Config {
    bool enabled{true};
    bool planar_revisit_initialization{false};
    double frequency_hz{1.0};
    double keyframe_distance{1.0}, keyframe_angle{0.2617993878}, keyframe_seconds{5.0};
    double minimum_loop_seconds{30.0}, descriptor_threshold{0.20};
    double voxel_size{0.3}, maximum_correspondence{1.5};
    double maximum_fitness{0.15}, minimum_overlap{0.5}, overlap_distance{0.3};
    double huber_delta{1.345};
    double vertical_sigma{0.10};
    double export_frame_interval{0.0};
    double trajectory_exclusion_radius{0.0};
    Eigen::Vector3d lidar_in_imu{Eigen::Vector3d::Zero()};
    std::size_t candidates{5}, preview_points{300000}, chunk_bytes{64U << 20};
  };
  struct Snapshot {
    MapArchive::Frames frames;
    MapArchive::Poses keyframes;
    std::size_t loop_count{0}, processed_keyframes{0};
    double last_cycle_ms{0};
    std::string error;
  };
  LoopClosureBackend(const std::string &directory, Config config, bool start_worker = true);
  ~LoopClosureBackend();
  LoopClosureBackend(const LoopClosureBackend &) = delete;
  LoopClosureBackend &operator=(const LoopClosureBackend &) = delete;
  void append(const Cloud &scan_in_imu, const Pose &odom_pose, double stamp);
  Snapshot snapshot() const;
  Cloud preview(const Snapshot &snapshot) const;
  bool save(const std::string &path, std::string *error,
            const std::function<bool()> &cancel = {}, Snapshot *saved_snapshot = nullptr);
  // Also used to drain the final pending keyframes when ending a mapping session.
  // Serialized with the worker; never called from the LIO callback.
  void processPending();
  std::string directory() const;
  bool discardSavedChunks();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace fast_lio
