#ifndef FAST_LIO__REFERENCE_ODOMETRY_HPP_
#define FAST_LIO__REFERENCE_ODOMETRY_HPP_

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fast_lio
{
namespace reference_odometry
{

/// 不依赖 ROS 消息的刚体位姿，便于单元测试会话重定位逻辑。
struct Pose
{
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
};

struct Update
{
  Pose reference_to_tracking;
  Eigen::Vector3d linear_velocity{Eigen::Vector3d::Zero()};
  double stamp_seconds{0.0};
};

enum class UpdateError
{
  kNone,
  kInvalidStamp,
  kInvalidPose,
  kNonMonotonicStamp,
};

inline bool finiteVector(const Eigen::Vector3d & value)
{
  return std::isfinite(value.x()) && std::isfinite(value.y()) &&
         std::isfinite(value.z());
}

inline bool normalizePose(Pose * pose)
{
  if (pose == nullptr || !finiteVector(pose->translation) ||
      !std::isfinite(pose->rotation.w()) ||
      !std::isfinite(pose->rotation.x()) ||
      !std::isfinite(pose->rotation.y()) ||
      !std::isfinite(pose->rotation.z()))
  {
    return false;
  }
  const double norm = pose->rotation.norm();
  if (!std::isfinite(norm) || norm < 1.0e-9 || std::abs(norm - 1.0) > 0.05)
  {
    return false;
  }
  pose->rotation.normalize();
  return true;
}

inline Pose compose(const Pose & parent_to_middle, const Pose & middle_to_child)
{
  Pose result;
  result.rotation = parent_to_middle.rotation * middle_to_child.rotation;
  result.rotation.normalize();
  result.translation = parent_to_middle.translation +
    parent_to_middle.rotation * middle_to_child.translation;
  return result;
}

inline Pose inverse(const Pose & value)
{
  Pose result;
  result.rotation = value.rotation.conjugate();
  result.translation = result.rotation * (-value.translation);
  return result;
}

/// 把仿真世界系中的绝对位姿转换成一次 SLAM 会话内的连续局部里程计。
///
/// 首个有效样本定义 reference 原点；后续速度由相邻相对位姿计算，避免依赖
/// 不同仿真器对 nav_msgs/Odometry.twist 所采用的坐标系约定。
class SessionTracker
{
public:
  void reset()
  {
    initialized_ = false;
    previous_stamp_seconds_ = 0.0;
    world_to_tracking_at_start_ = Pose{};
    previous_reference_to_tracking_ = Pose{};
  }

  bool update(
    Pose world_to_pose_frame,
    Pose pose_frame_to_tracking,
    const double stamp_seconds,
    Update * output,
    UpdateError * error = nullptr)
  {
    setError(error, UpdateError::kNone);
    if (output == nullptr || !std::isfinite(stamp_seconds) || stamp_seconds <= 0.0)
    {
      setError(error, UpdateError::kInvalidStamp);
      return false;
    }
    if (!normalizePose(&world_to_pose_frame) ||
        !normalizePose(&pose_frame_to_tracking))
    {
      setError(error, UpdateError::kInvalidPose);
      return false;
    }
    if (initialized_ && stamp_seconds <= previous_stamp_seconds_)
    {
      setError(error, UpdateError::kNonMonotonicStamp);
      return false;
    }

    const Pose world_to_tracking = compose(
      world_to_pose_frame, pose_frame_to_tracking);
    if (!initialized_)
    {
      world_to_tracking_at_start_ = world_to_tracking;
      previous_reference_to_tracking_ = Pose{};
      previous_stamp_seconds_ = stamp_seconds;
      initialized_ = true;
      output->reference_to_tracking = Pose{};
      output->linear_velocity.setZero();
      output->stamp_seconds = stamp_seconds;
      return true;
    }

    const Pose relative = compose(inverse(world_to_tracking_at_start_), world_to_tracking);
    const double delta_time = stamp_seconds - previous_stamp_seconds_;
    output->reference_to_tracking = relative;
    output->linear_velocity =
      (relative.translation - previous_reference_to_tracking_.translation) / delta_time;
    output->stamp_seconds = stamp_seconds;
    previous_reference_to_tracking_ = relative;
    previous_stamp_seconds_ = stamp_seconds;
    return true;
  }

  bool initialized() const
  {
    return initialized_;
  }

private:
  static void setError(UpdateError * output, const UpdateError value)
  {
    if (output != nullptr)
    {
      *output = value;
    }
  }

  bool initialized_{false};
  double previous_stamp_seconds_{0.0};
  Pose world_to_tracking_at_start_;
  Pose previous_reference_to_tracking_;
};

}  // namespace reference_odometry
}  // namespace fast_lio

#endif  // FAST_LIO__REFERENCE_ODOMETRY_HPP_
