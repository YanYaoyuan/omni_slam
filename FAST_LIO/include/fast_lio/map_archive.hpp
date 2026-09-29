#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace fast_lio {

// Dense scans are stored once in bounded disk chunks. All poses are IMU poses;
// the caller supplies scans already transformed from LiDAR to IMU coordinates.
// Loop optimization changes export/preview transforms, never these scans or LIO.
class MapArchive {
public:
  using Cloud = pcl::PointCloud<pcl::PointXYZINormal>;
  using Pose = Eigen::Isometry3d;
  using Poses = std::vector<Pose, Eigen::aligned_allocator<Pose>>;
  struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::uint64_t first_point{0}, point_count{0};
    std::size_t keyframe{0};
    double stamp{0};
    Pose odom_pose{Pose::Identity()}, keyframe_from_frame{Pose::Identity()};
  };
  using Frames = std::vector<Frame, Eigen::aligned_allocator<Frame>>;

  MapArchive(const std::string &parent, std::size_t chunk_bytes = 64U << 20,
             std::size_t preview_points = 300000);
  ~MapArchive();
  MapArchive(const MapArchive &) = delete;
  MapArchive &operator=(const MapArchive &) = delete;
  std::size_t append(const Cloud &scan, const Pose &odom_pose, double stamp,
                     std::size_t keyframe, const Pose &keyframe_from_frame);
  Cloud readFrame(std::size_t index) const;
  Frames frames() const;
  Cloud preview(const Poses &optimized_keyframes = {},
                std::size_t frame_count = std::numeric_limits<std::size_t>::max()) const;
  bool exportPcd(const std::string &path, const Poses &optimized_keyframes,
                 std::string *error, const std::function<bool()> &cancel = {},
                 std::size_t frame_count = std::numeric_limits<std::size_t>::max(),
                 double minimum_frame_interval = 0.0);
  std::uint64_t pointCount() const;
  std::string directory() const;
  // Explicit: only the owner may discard a successfully saved session.
  // Destruction alone leaves chunks available after a failed/canceled save.
  bool discardSavedChunks();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace fast_lio
