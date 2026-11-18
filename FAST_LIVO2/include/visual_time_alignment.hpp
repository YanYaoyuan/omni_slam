#pragma once
#include "common_lib.h"

namespace fast_livo {
inline M3D rotation_between(const deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu,
                           double begin,double end,const V3D &bias)
{
  if (imu.empty() || !std::isfinite(begin) || !std::isfinite(end) || end<begin)
    throw std::invalid_argument("invalid visual alignment interval");
  if (stamp_seconds(imu.front()->header.stamp)>begin+0.006)
    throw std::invalid_argument("visual alignment has no preceding IMU coverage");
  M3D rotation=M3D::Identity();
  for (size_t i=0;i<imu.size();++i) {
    const double t=stamp_seconds(imu[i]->header.stamp);
    const double stop=i+1<imu.size()?stamp_seconds(imu[i+1]->header.stamp):end;
    const double left=std::max(begin,t),right=std::min(end,stop);
    if (right<=left) continue;
    const auto &w=imu[i]->angular_velocity;
    V3D rate(w.x,w.y,w.z);
    if (i+1<imu.size() && stop>t) {
      const auto &next=imu[i+1]->angular_velocity;
      const V3D other(next.x,next.y,next.z);
      rate+=(other-rate)*((left+right)*0.5-t)/(stop-t);
    }
    const V3D unbiased=rate-bias;
    rotation=rotation*Exp(unbiased,right-left);
  }
  return rotation;
}
}
