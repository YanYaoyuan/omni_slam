#include "fast_lio/reference_odometry.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace fro = fast_lio::reference_odometry;

namespace
{

fro::Pose pose(
  const double x, const double y, const double z,
  const double yaw = 0.0)
{
  fro::Pose result;
  result.translation = Eigen::Vector3d(x, y, z);
  result.rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
  return result;
}

}  // namespace

TEST(ReferenceOdometry, FirstTrackingPoseDefinesSessionOrigin)
{
  fro::SessionTracker tracker;
  fro::Update update;

  // 外部消息描述 base，固定外参描述 base->tracking。无论世界绝对坐标为何，
  // 第一个 tracking 位姿都必须成为本次 SLAM 会话的局部原点。
  ASSERT_TRUE(tracker.update(pose(12.0, -4.0, 0.3, 0.6), pose(0.2, 0.0, 0.1), 10.0, &update));
  EXPECT_TRUE(update.reference_to_tracking.translation.isApprox(Eigen::Vector3d::Zero()));
  EXPECT_TRUE(update.reference_to_tracking.rotation.isApprox(Eigen::Quaterniond::Identity()));
  EXPECT_TRUE(update.linear_velocity.isApprox(Eigen::Vector3d::Zero()));
}

TEST(ReferenceOdometry, ProducesRelativePoseAndVelocity)
{
  fro::SessionTracker tracker;
  fro::Update update;
  ASSERT_TRUE(tracker.update(pose(5.0, 2.0, 0.0), pose(0.0, 0.0, 0.0), 20.0, &update));
  ASSERT_TRUE(tracker.update(pose(5.3, 1.8, 0.0, 0.2), pose(0.0, 0.0, 0.0), 20.1, &update));

  EXPECT_TRUE(update.reference_to_tracking.translation.isApprox(
    Eigen::Vector3d(0.3, -0.2, 0.0), 1.0e-9));
  EXPECT_TRUE(update.linear_velocity.isApprox(
    Eigen::Vector3d(3.0, -2.0, 0.0), 1.0e-9));
  EXPECT_NEAR(
    Eigen::AngleAxisd(update.reference_to_tracking.rotation).angle(), 0.2, 1.0e-9);
}

TEST(ReferenceOdometry, AppliesPoseFrameToTrackingExtrinsic)
{
  fro::SessionTracker tracker;
  fro::Update update;
  const fro::Pose base_to_tracking = pose(1.0, 0.0, 0.0);
  ASSERT_TRUE(tracker.update(pose(0.0, 0.0, 0.0), base_to_tracking, 30.0, &update));
  ASSERT_TRUE(tracker.update(
    pose(0.0, 0.0, 0.0, M_PI_2), base_to_tracking, 31.0, &update));

  // base 原地旋转时，偏置安装的 tracking frame 会绕初始 tracking 原点转动。
  EXPECT_TRUE(update.reference_to_tracking.translation.isApprox(
    Eigen::Vector3d(-1.0, 1.0, 0.0), 1.0e-9));
}

TEST(ReferenceOdometry, RejectsInvalidAndOutOfOrderSamplesWithoutCorruptingState)
{
  fro::SessionTracker tracker;
  fro::Update update;
  fro::UpdateError error = fro::UpdateError::kNone;
  ASSERT_TRUE(tracker.update(pose(0.0, 0.0, 0.0), pose(0.0, 0.0, 0.0), 40.0, &update));

  fro::Pose invalid = pose(1.0, 0.0, 0.0);
  invalid.rotation = Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0);
  EXPECT_FALSE(tracker.update(invalid, pose(0.0, 0.0, 0.0), 40.1, &update, &error));
  EXPECT_EQ(error, fro::UpdateError::kInvalidPose);
  EXPECT_FALSE(tracker.update(
    pose(1.0, 0.0, 0.0), pose(0.0, 0.0, 0.0), 39.9, &update, &error));
  EXPECT_EQ(error, fro::UpdateError::kNonMonotonicStamp);

  ASSERT_TRUE(tracker.update(pose(0.2, 0.0, 0.0), pose(0.0, 0.0, 0.0), 40.2, &update));
  EXPECT_TRUE(update.reference_to_tracking.translation.isApprox(
    Eigen::Vector3d(0.2, 0.0, 0.0), 1.0e-9));
  EXPECT_TRUE(update.linear_velocity.isApprox(
    Eigen::Vector3d(1.0, 0.0, 0.0), 1.0e-9));
}

TEST(ReferenceOdometry, ResetStartsANewSession)
{
  fro::SessionTracker tracker;
  fro::Update update;
  ASSERT_TRUE(tracker.update(pose(1.0, 0.0, 0.0), pose(0.0, 0.0, 0.0), 50.0, &update));
  ASSERT_TRUE(tracker.update(pose(2.0, 0.0, 0.0), pose(0.0, 0.0, 0.0), 51.0, &update));
  tracker.reset();
  ASSERT_TRUE(tracker.update(pose(8.0, 3.0, 0.0), pose(0.0, 0.0, 0.0), 52.0, &update));
  EXPECT_TRUE(update.reference_to_tracking.translation.isApprox(Eigen::Vector3d::Zero()));
  EXPECT_TRUE(update.linear_velocity.isApprox(Eigen::Vector3d::Zero()));
}
