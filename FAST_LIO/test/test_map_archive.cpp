#include "fast_lio/map_archive.hpp"
#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <filesystem>
#include <limits>
#include <thread>
#include <atomic>

namespace {
using Archive = fast_lio::MapArchive;
class ArchiveTest : public ::testing::Test {
protected:
  std::string root;
  void SetUp() override {
    char pattern[] = "/tmp/heng-map-archive-XXXXXX";
    root = ::mkdtemp(pattern);
  }
  void TearDown() override { std::filesystem::remove_all(root); }
  static Archive::Cloud scan(int n) {
    Archive::Cloud result;
    for (int i=0; i<n; ++i) {
      pcl::PointXYZINormal p{};
      p.x=float(i); p.y=2; p.z=3; p.intensity=20;
      p.normal_x=1; p.normal_y=0; p.normal_z=0; p.curvature=0.25;
      result.push_back(p);
    }
    return result;
  }
  static Archive::Pose pose(double x) {
    auto p=Archive::Pose::Identity(); p.translation().x()=x; return p;
  }
};

TEST_F(ArchiveTest, PreservesDenseRecordsAcrossBoundedChunks) {
  Archive archive(root, 128, 5);
  auto input=scan(17);
  auto bad=input.front(); bad.x=std::numeric_limits<float>::quiet_NaN(); input.push_back(bad);
  archive.append(input, pose(2), 1.0, 0, pose(0));
  archive.append(scan(3), pose(3), 2.0, 0, pose(1));
  ASSERT_EQ(archive.pointCount(), 20u);
  ASSERT_EQ(archive.frames().size(), 2u);
  ASSERT_EQ(archive.readFrame(0).size(), 17u);
  EXPECT_EQ(archive.readFrame(0)[16].x, 16.0f);
  EXPECT_EQ(archive.readFrame(1)[2].curvature, 0.25f);
  for (const auto &entry: std::filesystem::directory_iterator(archive.directory()))
    EXPECT_LE(entry.file_size(), 128u);
  EXPECT_EQ(archive.preview().size(), 5u);
  std::string error;
  ASSERT_TRUE(archive.exportPcd(root+"/map.pcd", {}, &error)) << error;
  Archive::Cloud loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(root+"/map.pcd", loaded), 0);
  ASSERT_EQ(loaded.size(), 20u);
  EXPECT_EQ(loaded.front().x, 2.0f);
  EXPECT_EQ(loaded.back().x, 5.0f);
  EXPECT_TRUE(archive.discardSavedChunks());
}

TEST_F(ArchiveTest, LoopCorrectionTransformsEachFrameWithoutChangingStoredScans) {
  Archive archive(root, 128, 20);
  archive.append(scan(1), pose(2), 1.0, 0, pose(0));
  archive.append(scan(1), pose(3), 2.0, 0, pose(1));
  Archive::Poses optimized{pose(10)};
  optimized[0].linear() = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  std::string error;
  ASSERT_TRUE(archive.exportPcd(root+"/corrected.pcd", optimized, &error)) << error;
  Archive::Cloud loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(root+"/corrected.pcd", loaded), 0);
  ASSERT_EQ(loaded.size(), 2u);
  EXPECT_NEAR(loaded[0].x, 8, 1e-5);
  EXPECT_NEAR(loaded[1].x, 8, 1e-5);
  EXPECT_NEAR(loaded[1].y, 1, 1e-5);
  EXPECT_NEAR(loaded[1].normal_y, 1, 1e-5);
  EXPECT_NEAR(archive.preview(optimized)[1].y, 1, 1e-5);
  EXPECT_EQ(archive.readFrame(1)[0].x, 0);
  EXPECT_EQ(archive.frames()[1].odom_pose.translation().x(), 3);
}

TEST_F(ArchiveTest, ExportCanSelectOneDenseFramePerSecondAndKeepFinalFrame) {
  Archive archive(root, 4096, 20);
  for (int i=0;i<7;++i) archive.append(scan(2),pose(i),i*.2,0,pose(i));
  std::string error;
  ASSERT_TRUE(archive.exportPcd(root+"/sampled.pcd",{},&error,{},7,1.0))<<error;
  Archive::Cloud loaded;
  ASSERT_EQ(pcl::io::loadPCDFile(root+"/sampled.pcd",loaded),0);
  // t=0、t=1.0，以及不足下一秒但必须保留的最终帧 t=1.2。
  EXPECT_EQ(loaded.size(),6u);
  EXPECT_TRUE(archive.discardSavedChunks());
}

TEST_F(ArchiveTest, CancellationRetainsPreviousMapAndUnsavedChunks) {
  Archive archive(root, 128, 5);
  const auto target=root+"/map.pcd";
  { std::ofstream f(target); f << "previous map"; }
  archive.append(scan(17), pose(2), 1, 0, pose(0));
  std::string error;
  EXPECT_FALSE(archive.exportPcd(target, {}, &error, [] { return true; }));
  EXPECT_NE(error.find("canceled"), std::string::npos);
  std::ifstream input(target);
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "previous map");
  EXPECT_FALSE(archive.discardSavedChunks());
  for (const auto &entry: std::filesystem::directory_iterator(root))
    EXPECT_EQ(entry.path().filename().string().find(".tmp."), std::string::npos);
  ASSERT_TRUE(archive.exportPcd(target, {}, &error));
  archive.append(scan(1), pose(3), 2, 1, pose(0));
  EXPECT_FALSE(archive.discardSavedChunks());
}

TEST_F(ArchiveTest, ConcurrentPreviewOnlySeesCompleteFrameSnapshots) {
  Archive archive(root, 4096, 50);
  std::atomic<bool> finished{false};
  std::thread reader([&] {
    while (!finished.load()) {
      auto preview=archive.preview();
      EXPECT_LE(preview.size(), 50u);
      for (const auto &p: preview) EXPECT_TRUE(std::isfinite(p.x));
    }
  });
  for (int i=0; i<100; ++i) archive.append(scan(100), pose(i), i, 0, pose(i));
  finished.store(true); reader.join();
  EXPECT_EQ(archive.pointCount(), 10000u);
}
}
