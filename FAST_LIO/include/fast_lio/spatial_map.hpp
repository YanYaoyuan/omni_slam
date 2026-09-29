#pragma once

#include "fast_lio/map_archive.hpp"
#include <array>
#include <functional>
#include <memory>
#include <string>

namespace fast_lio {
// Streams legacy binary, binary_compressed and ASCII PCDs into a reusable,
// voxel-filtered disk index. Dense maps are never loaded as one PCL cloud.
class SpatialMap {
public:
  using Cloud=MapArchive::Cloud;
  using Tile=std::array<std::int64_t,3>;
  struct Config {
    double voxel_size{0.3}, tile_size{20.0};
    std::size_t cache_points{1000000}, region_points{2000000}, preview_points{300000};
  };
  struct Stats {
    std::uint64_t input_points{0}, valid_points{0}, filtered_points{0};
    std::size_t tiles{0}, cached_points{0};
    bool reused_index{false};
    double tile_size{0};
  };
  SpatialMap(const std::string &pcd, const std::string &cache_directory, Config config,
             const std::function<bool()> &cancel = {});
  ~SpatialMap();
  SpatialMap(const SpatialMap &)=delete;
  SpatialMap &operator=(const SpatialMap &)=delete;
  std::vector<Tile> tilesFor(const Eigen::Vector3d &center, double radius) const;
  // Returned clouds are immutable and may outlive eviction from the bounded LRU.
  Cloud::ConstPtr tile(const Tile &key);
  std::uint64_t pointCount(const Tile &key) const;
  Cloud region(const Eigen::Vector3d &center, double radius);
  Cloud preview() const;
  Stats stats() const;
  std::string directory() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace fast_lio
