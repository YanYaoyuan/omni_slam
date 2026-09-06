#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Geometry>

namespace icp_relocalization
{

/** 单帧 ICP 候选经过质量与位姿一致性门后的判定。 */
enum class ConvergenceDecision
{
  kRejected,
  kAccumulating,
  kClusterRestarted,
  kReady,
};

/**
 * @brief 对连续 ICP 候选建立稳定的 SE(3) 收敛证据。
 *
 * 单看 fitness 连续低于阈值并不能证明定位稳定：重复结构或每帧点云覆盖变化可能
 * 产生一组分数都很好、但位置互相漂动的解。本类要求候选同时落在同一个平移和
 * 旋转簇内，并发布该簇的均值位姿，从而避免把最后一帧抖动直接写入 TF 树。
 *
 * 输入质量是否合格由调用方根据 ICP hasConverged/fitness 判定；本类只负责有限值、
 * 刚体变换和跨帧一致性。它不依赖 ROS/PCL，便于在主机与目标板使用同一组测试。
 */
class RelocalizationConvergenceGate
{
public:
  RelocalizationConvergenceGate(
    const std::size_t required_samples,
    const double position_tolerance_m,
    const double angle_tolerance_rad)
  : required_samples_(required_samples),
    position_tolerance_m_(position_tolerance_m),
    angle_tolerance_rad_(angle_tolerance_rad)
  {
    if (required_samples_ == 0U || !std::isfinite(position_tolerance_m_) ||
      !std::isfinite(angle_tolerance_rad_) || position_tolerance_m_ <= 0.0 ||
      angle_tolerance_rad_ <= 0.0 || angle_tolerance_rad_ > M_PI)
    {
      throw std::invalid_argument("invalid relocalization convergence gate limits");
    }
  }

  ConvergenceDecision Observe(
    const bool quality_accepted,
    const Eigen::Matrix4f & candidate)
  {
    Eigen::Vector3d translation;
    Eigen::Quaterniond orientation;
    if (!quality_accepted ||
      !DecodeRigidTransform(candidate, translation, orientation))
    {
      Reset();
      return ConvergenceDecision::kRejected;
    }

    if (sample_count_ == 0U) {
      StartCluster(translation, orientation);
      return required_samples_ == 1U ?
             ConvergenceDecision::kReady :
             ConvergenceDecision::kAccumulating;
    }

    const double position_delta = (translation - mean_translation_).norm();
    const double angle_delta = AngularDistance(orientation, mean_orientation_);
    if (position_delta > position_tolerance_m_ ||
      angle_delta > angle_tolerance_rad_)
    {
      // 不一致候选不能简单丢弃后继续累计旧簇，否则交替出现的两个局部极小值
      // 仍可能凑够次数。新候选成为新簇的第一个样本，旧证据全部作废。
      StartCluster(translation, orientation);
      return required_samples_ == 1U ?
             ConvergenceDecision::kReady :
             ConvergenceDecision::kClusterRestarted;
    }

    ++sample_count_;
    const double weight = 1.0 / static_cast<double>(sample_count_);
    mean_translation_ += weight * (translation - mean_translation_);
    mean_orientation_ = mean_orientation_.slerp(weight, orientation).normalized();
    return sample_count_ >= required_samples_ ?
           ConvergenceDecision::kReady :
           ConvergenceDecision::kAccumulating;
  }

  void Reset() noexcept
  {
    sample_count_ = 0U;
    mean_translation_.setZero();
    mean_orientation_.setIdentity();
  }

  std::size_t sample_count() const noexcept {return sample_count_;}
  std::size_t required_samples() const noexcept {return required_samples_;}

  /** 返回当前稳定簇的均值刚体变换；尚无候选时返回单位矩阵。 */
  Eigen::Matrix4f Estimate() const
  {
    Eigen::Matrix4f estimate = Eigen::Matrix4f::Identity();
    if (sample_count_ == 0U) {
      return estimate;
    }
    estimate.block<3, 3>(0, 0) =
      mean_orientation_.toRotationMatrix().cast<float>();
    estimate.block<3, 1>(0, 3) = mean_translation_.cast<float>();
    return estimate;
  }

private:
  static bool DecodeRigidTransform(
    const Eigen::Matrix4f & candidate,
    Eigen::Vector3d & translation,
    Eigen::Quaterniond & orientation)
  {
    if (!candidate.allFinite()) {
      return false;
    }
    const Eigen::Matrix3d rotation =
      candidate.block<3, 3>(0, 0).cast<double>();
    const double determinant = rotation.determinant();
    if (!std::isfinite(determinant) || std::abs(determinant - 1.0) > 0.05) {
      return false;
    }
    orientation = Eigen::Quaterniond(rotation);
    if (!orientation.coeffs().allFinite() || orientation.norm() < 1e-9) {
      return false;
    }
    orientation.normalize();
    translation = candidate.block<3, 1>(0, 3).cast<double>();
    return translation.allFinite();
  }

  static double AngularDistance(
    const Eigen::Quaterniond & lhs,
    const Eigen::Quaterniond & rhs)
  {
    // q 与 -q 表示同一旋转，取绝对内积后结果自然跨越 ±pi yaw 边界。
    const double dot = std::abs(lhs.dot(rhs));
    return 2.0 * std::acos(std::max(0.0, std::min(1.0, dot)));
  }

  void StartCluster(
    const Eigen::Vector3d & translation,
    const Eigen::Quaterniond & orientation)
  {
    sample_count_ = 1U;
    mean_translation_ = translation;
    mean_orientation_ = orientation;
  }

  std::size_t required_samples_;
  double position_tolerance_m_;
  double angle_tolerance_rad_;
  std::size_t sample_count_{0U};
  Eigen::Vector3d mean_translation_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond mean_orientation_{Eigen::Quaterniond::Identity()};
};

}  // namespace icp_relocalization
