#include "fast_lio/extent_safe_voxel_filter.hpp"

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include <limits>
#include <stdexcept>

namespace
{

TEST(ExtentSafeVoxelFilter, DownsamplesCloudWhoseGlobalVoxelGridWouldOverflow)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back(pcl::PointXYZ{-4000.10F, -700.10F, -150.10F});
  cloud.push_back(pcl::PointXYZ{-4000.20F, -700.20F, -150.20F});
  cloud.push_back(pcl::PointXYZ{3400.10F, 3900.10F, 1100.10F});
  cloud.push_back(pcl::PointXYZ{3400.20F, 3900.20F, 1100.20F});

  const auto stats = fast_lio::pointcloud::extentSafeVoxelFilter(cloud, 0.5, &cloud);

  EXPECT_EQ(stats.input_points, 4U);
  EXPECT_EQ(stats.output_points, 2U);
  EXPECT_EQ(cloud.size(), 2U);
  EXPECT_EQ(stats.rejected_non_finite, 0U);
  EXPECT_EQ(stats.rejected_index_overflow, 0U);
}

TEST(ExtentSafeVoxelFilter, KeepsPointNearestToVoxelCenterAndExtendedFields)
{
  pcl::PointCloud<pcl::PointXYZINormal> cloud;
  pcl::PointXYZINormal far_from_center;
  far_from_center.x = 0.01F;
  far_from_center.y = 0.01F;
  far_from_center.z = 0.01F;
  far_from_center.intensity = 10.0F;
  pcl::PointXYZINormal near_center;
  near_center.x = 0.49F;
  near_center.y = 0.49F;
  near_center.z = 0.49F;
  near_center.intensity = 42.0F;
  cloud.push_back(far_from_center);
  cloud.push_back(near_center);

  pcl::PointCloud<pcl::PointXYZINormal> output;
  const auto stats = fast_lio::pointcloud::extentSafeVoxelFilter(cloud, 1.0, &output);

  ASSERT_EQ(stats.output_points, 1U);
  ASSERT_EQ(output.size(), 1U);
  EXPECT_FLOAT_EQ(output.front().x, 0.49F);
  EXPECT_FLOAT_EQ(output.front().intensity, 42.0F);
}

TEST(ExtentSafeVoxelFilter, RejectsInvalidPointsAndParameters)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back(pcl::PointXYZ{1.0F, 2.0F, 3.0F});
  cloud.push_back(
    pcl::PointXYZ{std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F});

  pcl::PointCloud<pcl::PointXYZ> output;
  const auto stats = fast_lio::pointcloud::extentSafeVoxelFilter(cloud, 0.5, &output);
  EXPECT_EQ(stats.output_points, 1U);
  EXPECT_EQ(stats.rejected_non_finite, 1U);
  EXPECT_THROW(
    fast_lio::pointcloud::extentSafeVoxelFilter(cloud, 0.0, &output),
    std::invalid_argument);
  EXPECT_THROW(
    fast_lio::pointcloud::extentSafeVoxelFilter<pcl::PointXYZ>(cloud, 0.5, nullptr),
    std::invalid_argument);
}

TEST(ExtentSafeVoxelFilter, RemovesNonFiniteAndOutOfSensorRangePoints)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back(pcl::PointXYZ{3.0F, 4.0F, 0.0F});
  cloud.push_back(pcl::PointXYZ{60.0F, 80.0F, 0.0F});
  cloud.push_back(pcl::PointXYZ{60.1F, 80.0F, 0.0F});
  cloud.push_back(
    pcl::PointXYZ{0.0F, std::numeric_limits<float>::infinity(), 0.0F});

  const auto stats = fast_lio::pointcloud::filterByMaximumRange(cloud, 100.0, &cloud);

  ASSERT_EQ(cloud.size(), 2U);
  EXPECT_EQ(stats.input_points, 4U);
  EXPECT_EQ(stats.output_points, 2U);
  EXPECT_EQ(stats.rejected_out_of_range, 1U);
  EXPECT_EQ(stats.rejected_non_finite, 1U);
  EXPECT_THROW(
    fast_lio::pointcloud::filterByMaximumRange(cloud, -1.0, &cloud),
    std::invalid_argument);
}

}  // namespace
