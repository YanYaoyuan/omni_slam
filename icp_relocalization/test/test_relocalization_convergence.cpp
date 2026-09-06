#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "icp_relocalization/relocalization_convergence.hpp"

namespace
{

Eigen::Matrix4f Pose(const float x, const float y, const float yaw)
{
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3, 3>(0, 0) =
    Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  pose(0, 3) = x;
  pose(1, 3) = y;
  return pose;
}

TEST(RelocalizationConvergenceGate, RequiresOneConsistentPoseCluster)
{
  icp_relocalization::RelocalizationConvergenceGate gate(3U, 0.30, 0.20);
  EXPECT_EQ(
    gate.Observe(true, Pose(1.00F, 2.00F, 0.10F)),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(true, Pose(1.10F, 1.95F, 0.12F)),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.95F, 2.05F, 0.08F)),
    icp_relocalization::ConvergenceDecision::kReady);
  EXPECT_EQ(gate.sample_count(), 3U);
  EXPECT_NEAR(gate.Estimate()(0, 3), 1.0167F, 1e-3F);
  EXPECT_NEAR(gate.Estimate()(1, 3), 2.0F, 1e-3F);
}

TEST(RelocalizationConvergenceGate, InconsistentCandidateRestartsEvidence)
{
  icp_relocalization::RelocalizationConvergenceGate gate(3U, 0.25, 0.20);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.0F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.1F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(true, Pose(1.0F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kClusterRestarted);
  EXPECT_EQ(gate.sample_count(), 1U);
  EXPECT_EQ(
    gate.Observe(true, Pose(1.1F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.9F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kReady);
}

TEST(RelocalizationConvergenceGate, RejectedOrInvalidCandidateClearsEvidence)
{
  icp_relocalization::RelocalizationConvergenceGate gate(2U, 0.30, 0.20);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.0F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(false, Pose(0.0F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kRejected);
  EXPECT_EQ(gate.sample_count(), 0U);

  Eigen::Matrix4f invalid = Pose(0.0F, 0.0F, 0.0F);
  invalid(0, 3) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(
    gate.Observe(true, invalid),
    icp_relocalization::ConvergenceDecision::kRejected);
  EXPECT_EQ(gate.sample_count(), 0U);
}

TEST(RelocalizationConvergenceGate, QuaternionSignAndYawWrapAreContinuous)
{
  icp_relocalization::RelocalizationConvergenceGate gate(2U, 0.20, 0.10);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.0F, 0.0F, static_cast<float>(M_PI - 0.02))),
    icp_relocalization::ConvergenceDecision::kAccumulating);
  EXPECT_EQ(
    gate.Observe(true, Pose(0.0F, 0.0F, static_cast<float>(-M_PI + 0.02))),
    icp_relocalization::ConvergenceDecision::kReady);
}

}  // namespace
