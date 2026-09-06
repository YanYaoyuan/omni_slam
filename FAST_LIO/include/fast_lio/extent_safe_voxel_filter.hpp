#pragma once

#include <pcl/point_cloud.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fast_lio
{
namespace pointcloud
{

/**
 * @brief 点云体素降采样统计信息。
 *
 * rejected_non_finite 用于识别 PCD 中的 NaN/Inf；rejected_index_overflow
 * 表示坐标无法安全换算成 64 位体素索引。调用方应把这两个字段写入日志，
 * 便于区分地图质量问题和算法本身的定位问题。
 */
struct VoxelFilterStats
{
  std::size_t input_points{0U};
  std::size_t output_points{0U};
  std::size_t rejected_non_finite{0U};
  std::size_t rejected_index_overflow{0U};
};

struct RangeFilterStats
{
  std::size_t input_points{0U};
  std::size_t output_points{0U};
  std::size_t rejected_non_finite{0U};
  std::size_t rejected_out_of_range{0U};
};

namespace detail
{

struct VoxelKey
{
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const VoxelKey & other) const noexcept
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const noexcept
  {
    // 分别散列三个有符号 64 位索引，避免把体素范围压回 32 位。
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + static_cast<std::size_t>(0x9e3779b9U) +
      (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + static_cast<std::size_t>(0x9e3779b9U) +
      (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

inline bool coordinateToVoxelIndex(
  const float coordinate,
  const double inverse_leaf_size,
  std::int64_t * const index)
{
  if (!std::isfinite(coordinate) || index == nullptr) {
    return false;
  }

  const double scaled = std::floor(static_cast<double>(coordinate) * inverse_leaf_size);
  constexpr double kMinIndex = static_cast<double>(std::numeric_limits<std::int64_t>::min());
  constexpr double kMaxIndex = static_cast<double>(std::numeric_limits<std::int64_t>::max());
  if (!std::isfinite(scaled) || scaled < kMinIndex || scaled >= kMaxIndex) {
    return false;
  }
  *index = static_cast<std::int64_t>(scaled);
  return true;
}

}  // namespace detail

/**
 * @brief 对任意空间跨度的点云执行确定性的体素降采样。
 *
 * PCL VoxelGrid 使用三轴体素数量的 32 位乘积。少量公里级离群点就可能让
 * 乘积溢出，随后过滤器会原样返回整张地图。这里用三维 64 位体素键建立
 * 哈希表，不依赖整体包围盒大小；同一体素保留最靠近体素中心的真实点，
 * 因此不会合成不存在的表面，也能保留 intensity/normal 等扩展字段。
 *
 * input 与 output 可以指向同一个点云，函数在末尾才替换输出。
 *
 * @throws std::invalid_argument 当叶大小无效或 output 为空时抛出。
 */
template<typename PointT>
VoxelFilterStats extentSafeVoxelFilter(
  const pcl::PointCloud<PointT> & input,
  const double leaf_size,
  pcl::PointCloud<PointT> * const output)
{
  if (output == nullptr) {
    throw std::invalid_argument("voxel filter output must not be null");
  }
  if (!std::isfinite(leaf_size) || leaf_size <= 0.0) {
    throw std::invalid_argument("voxel leaf size must be finite and positive");
  }

  VoxelFilterStats stats;
  stats.input_points = input.size();

  pcl::PointCloud<PointT> filtered;
  filtered.header = input.header;
  filtered.sensor_origin_ = input.sensor_origin_;
  filtered.sensor_orientation_ = input.sensor_orientation_;
  filtered.reserve(input.size());

  std::unordered_map<detail::VoxelKey, std::size_t, detail::VoxelKeyHash> voxel_slots;
  voxel_slots.reserve(input.size());
  std::vector<double> best_center_distance_squared;
  best_center_distance_squared.reserve(input.size());

  const double inverse_leaf_size = 1.0 / leaf_size;
  for (const auto & point : input.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      ++stats.rejected_non_finite;
      continue;
    }

    detail::VoxelKey key;
    if (!detail::coordinateToVoxelIndex(point.x, inverse_leaf_size, &key.x) ||
      !detail::coordinateToVoxelIndex(point.y, inverse_leaf_size, &key.y) ||
      !detail::coordinateToVoxelIndex(point.z, inverse_leaf_size, &key.z))
    {
      ++stats.rejected_index_overflow;
      continue;
    }

    const double center_x = (static_cast<double>(key.x) + 0.5) * leaf_size;
    const double center_y = (static_cast<double>(key.y) + 0.5) * leaf_size;
    const double center_z = (static_cast<double>(key.z) + 0.5) * leaf_size;
    const double delta_x = static_cast<double>(point.x) - center_x;
    const double delta_y = static_cast<double>(point.y) - center_y;
    const double delta_z = static_cast<double>(point.z) - center_z;
    const double center_distance_squared =
      delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    const auto inserted = voxel_slots.emplace(key, filtered.size());
    if (inserted.second) {
      filtered.push_back(point);
      best_center_distance_squared.push_back(center_distance_squared);
      continue;
    }

    const std::size_t slot = inserted.first->second;
    if (center_distance_squared < best_center_distance_squared[slot]) {
      filtered[slot] = point;
      best_center_distance_squared[slot] = center_distance_squared;
    }
  }

  filtered.width = static_cast<std::uint32_t>(filtered.size());
  filtered.height = 1U;
  filtered.is_dense = true;
  stats.output_points = filtered.size();
  *output = std::move(filtered);
  return stats;
}

/**
 * @brief 移除超出传感器可靠探测距离的点。
 *
 * FAST-LIO 的 det_range 不只是局部地图滑窗参数，也是当前扫描参与匹配的
 * 物理上限。若把仿真或驱动产生的公里级异常回波送入 KD 树最近邻搜索，
 * 单帧计算可能超过状态机失联门限。该过滤器在进入体素和匹配前统一清理
 * 非有限点及超距点，并保留点类型的所有扩展字段。
 */
template<typename PointT>
RangeFilterStats filterByMaximumRange(
  const pcl::PointCloud<PointT> & input,
  const double maximum_range,
  pcl::PointCloud<PointT> * const output)
{
  if (output == nullptr) {
    throw std::invalid_argument("range filter output must not be null");
  }
  if (!std::isfinite(maximum_range) || maximum_range <= 0.0) {
    throw std::invalid_argument("maximum range must be finite and positive");
  }

  RangeFilterStats stats;
  stats.input_points = input.size();

  pcl::PointCloud<PointT> filtered;
  filtered.header = input.header;
  filtered.sensor_origin_ = input.sensor_origin_;
  filtered.sensor_orientation_ = input.sensor_orientation_;
  filtered.reserve(input.size());

  const double maximum_range_squared = maximum_range * maximum_range;
  for (const auto & point : input.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      ++stats.rejected_non_finite;
      continue;
    }
    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    const double z = static_cast<double>(point.z);
    if (x * x + y * y + z * z > maximum_range_squared) {
      ++stats.rejected_out_of_range;
      continue;
    }
    filtered.push_back(point);
  }

  filtered.width = static_cast<std::uint32_t>(filtered.size());
  filtered.height = 1U;
  filtered.is_dense = true;
  stats.output_points = filtered.size();
  *output = std::move(filtered);
  return stats;
}

}  // namespace pointcloud
}  // namespace fast_lio
