#pragma once

#include "fast_lio/extent_safe_voxel_filter.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace fast_lio {
// Union of capsules swept by the LiDAR origin. This is geometric map cleaning,
// not a dynamic-object classifier. Coordinates must share one frame.
class TrajectoryExclusion {
public:
  explicit TrajectoryExclusion(double radius=0):radius_(radius) {
    if (!std::isfinite(radius) || radius<0) throw std::invalid_argument("invalid trajectory exclusion radius");
  }
  double radius() const { return radius_; }
  static double squaredDistance(const Eigen::Vector3d &p,const Eigen::Vector3d &a,const Eigen::Vector3d &b) {
    const Eigen::Vector3d d=b-a;
    const double t=d.squaredNorm()>0 ? std::max(0.0,std::min(1.0,(p-a).dot(d)/d.squaredNorm())):0;
    return (p-a-t*d).squaredNorm();
  }
  void add(const Eigen::Vector3d &p) {
    if (radius_<=0) return;
    if (!p.allFinite()) throw std::invalid_argument("non-finite trajectory position");
    const Eigen::Vector3d a=segments_.empty()?p:segments_.back().b;
    if (!segments_.empty() && (p-a).squaredNorm()<1e-12) return;
    // Short capsule pieces keep index creation linear even for long segments.
    const auto steps=std::max<std::size_t>(1,std::ceil((p-a).norm()/radius_));
    for (std::size_t i=0;i<steps;++i) {
      Segment s{a+(p-a)*(double(i)/steps),a+(p-a)*(double(i+1)/steps)};
      const auto id=segments_.size(); segments_.push_back(s);
      const auto lo=key(s.a.cwiseMin(s.b)-Eigen::Vector3d::Constant(radius_));
      const auto hi=key(s.a.cwiseMax(s.b)+Eigen::Vector3d::Constant(radius_));
      for(auto x=lo.x;x<=hi.x;++x) for(auto y=lo.y;y<=hi.y;++y) for(auto z=lo.z;z<=hi.z;++z)
        cells_[{x,y,z}].push_back(id);
    }
  }
  bool contains(const Eigen::Vector3d &p) const {
    if (radius_<=0 || !p.allFinite()) return false;
    const auto it=cells_.find(key(p)); if(it==cells_.end()) return false;
    for(auto id:it->second) {
      const auto &s=segments_[id];
      if(squaredDistance(p,s.a,s.b)<=radius_*radius_+1e-12) return true;
    }
    return false;
  }
private:
  using Key=pointcloud::detail::VoxelKey;
  struct Segment { Eigen::Vector3d a,b; };
  Key key(const Eigen::Vector3d &p) const {
    return {static_cast<std::int64_t>(std::floor(p.x()/(2*radius_))),
            static_cast<std::int64_t>(std::floor(p.y()/(2*radius_))),
            static_cast<std::int64_t>(std::floor(p.z()/(2*radius_)))};
  }
  double radius_;
  std::vector<Segment> segments_;
  std::unordered_map<Key,std::vector<std::size_t>,pointcloud::detail::VoxelKeyHash> cells_;
};
}  // namespace fast_lio
