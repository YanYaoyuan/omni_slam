#pragma once
#include "utils/types.h"
#include <cmath>
#include <stdexcept>

namespace fast_livo {
// Both input offsets and cut_ms are relative to the previous update time.
// Future offsets are rebased to the new update time. This must also be applied
// to retained points when the camera runs faster than the LiDAR scan rate.
inline void split_remaining_scan(const PointCloudXYZI &remaining, double cut_ms,
                                 PointCloudXYZI &current, PointCloudXYZI &future)
{
  if (!std::isfinite(cut_ms) || cut_ms<=0)
    throw std::invalid_argument("scan cut interval must be finite and positive");
  current.clear();future.clear();
  current.reserve(remaining.size());future.reserve(remaining.size());
  for (auto point:remaining) {
    // Epoch-time subtraction and conversion to float can leave a boundary
    // point a fraction of a microsecond below zero. Clamp only that numerical
    // residue, and continue rejecting actual negative acquisition offsets.
    if (!std::isfinite(point.curvature) || point.curvature < -0.001f)
      throw std::invalid_argument("retained point has invalid relative time");
    point.curvature=std::max(0.0f,point.curvature);
    if (point.curvature<cut_ms) current.push_back(point);
    else {
      point.curvature=static_cast<float>(point.curvature-cut_ms);
      future.push_back(point);
    }
  }
}
}
