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

namespace
{

TEST(RelocalizationConvergenceGate, CovarianceIsNeverAllZero)
{
  icp_relocalization::RelocalizationConvergenceGate gate(1U, 0.30, 0.20);
  ASSERT_EQ(
    gate.Observe(true, Pose(1.0F, 2.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kReady);
  const auto covariance = gate.Covariance(0.0);
  for (const std::size_t index : {0U, 7U, 14U}) {
    EXPECT_NEAR(covariance[index], 0.05 * 0.05, 1e-12);
  }
  for (const std::size_t index : {21U, 28U, 35U}) {
    EXPECT_NEAR(covariance[index], 0.02 * 0.02, 1e-12);
  }
  EXPECT_DOUBLE_EQ(covariance[1], 0.0);
  EXPECT_DOUBLE_EQ(covariance[6], 0.0);
}

TEST(RelocalizationConvergenceGate, CovarianceUsesFitnessAndClusterDispersion)
{
  icp_relocalization::RelocalizationConvergenceGate gate(3U, 0.50, 0.30);
  gate.Observe(true, Pose(0.0F, 0.0F, 0.00F));
  gate.Observe(true, Pose(0.2F, 0.0F, 0.10F));
  ASSERT_EQ(
    gate.Observe(true, Pose(0.4F, 0.0F, 0.20F)),
    icp_relocalization::ConvergenceDecision::kReady);
  const auto covariance = gate.Covariance(0.01);
  // x spreads 0, 0.2, 0.4 around its mean: sample variance 0.04.
  EXPECT_NEAR(covariance[0], 0.04, 1e-6);
  // y/z do not spread, so the ICP residual (0.01 m^2) is the floor.
  EXPECT_NEAR(covariance[7], 0.01, 1e-12);
  EXPECT_NEAR(covariance[14], 0.01, 1e-12);
  // yaw spreads about 0.1 rad around its mean.
  EXPECT_NEAR(covariance[35], 0.01, 1e-3);
  EXPECT_GT(covariance[35], 0.02 * 0.02);
  // Non-finite fitness falls back to the minimum floor instead of NaN.
  const auto non_finite = gate.Covariance(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(std::isfinite(non_finite[7]));
  EXPECT_NEAR(non_finite[7], 0.05 * 0.05, 1e-12);
}

TEST(RelocalizationConvergenceGate, ClusterRestartDropsOldDispersion)
{
  icp_relocalization::RelocalizationConvergenceGate gate(2U, 0.25, 0.20);
  gate.Observe(true, Pose(0.0F, 0.0F, 0.0F));
  gate.Observe(true, Pose(5.0F, 0.0F, 0.0F));  // restarts the cluster
  ASSERT_EQ(
    gate.Observe(true, Pose(5.0F, 0.0F, 0.0F)),
    icp_relocalization::ConvergenceDecision::kReady);
  EXPECT_NEAR(gate.Covariance(0.0)[0], 0.05 * 0.05, 1e-12);
}

}  // namespace
