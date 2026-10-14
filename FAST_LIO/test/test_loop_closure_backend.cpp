#include "fast_lio/loop_closure_backend.hpp"
#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <random>

namespace {
using Backend=fast_lio::LoopClosureBackend;
using Pose=Backend::Pose;
class LoopTest: public ::testing::Test {
protected:
  std::string root;
  void SetUp() override { char pattern[]="/tmp/heng-loop-XXXXXX"; root=::mkdtemp(pattern); }
  void TearDown() override { std::filesystem::remove_all(root); }
  static Pose pose(double x,double y=0,double yaw=0) {
    Pose p=Pose::Identity(); p.translation()<<x,y,0;
    p.linear()=Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ()).toRotationMatrix(); return p;
  }
  static Pose poseWithZ(double x,double y,double z,double yaw=0) {
    Pose p=pose(x,y,yaw); p.translation().z()=z; return p;
  }
  static Backend::Cloud scene(const Pose &sensor=Pose::Identity()) {
    // Fixed asymmetric scene, many non-coplanar points over 360 degrees.
    std::mt19937 random(71); std::uniform_real_distribution<float> range(-8,8), height(-1,3);
    Backend::Cloud cloud;
    for (int i=0;i<1800;++i) {
      const Eigen::Vector3d world(range(random),range(random),height(random));
      const auto local=sensor.inverse()*world;
      pcl::PointXYZINormal p{}; p.x=local.x(); p.y=local.y(); p.z=local.z(); p.intensity=i%255;
      cloud.push_back(p);
    }
    return cloud;
  }
};
TEST_F(LoopTest, PreservesOdomAndEveryScanUntilBackgroundOptimization) {
  Backend backend(root,Backend::Config{},false);
  for (int i=0;i<8;++i) backend.append(scene(pose(i*.2)),pose(i*.2),i*.2);
  const auto before=backend.snapshot();
  ASSERT_EQ(before.frames.size(),8u);
  EXPECT_EQ(before.keyframes.size(),2u);
  EXPECT_EQ(before.processed_keyframes,0u);
  backend.processPending();
  const auto after=backend.snapshot();
  EXPECT_TRUE(after.error.empty())<<after.error;
  EXPECT_EQ(after.processed_keyframes,2u); EXPECT_EQ(after.loop_count,0u);
  for (std::size_t i=0;i<before.frames.size();++i)
    EXPECT_TRUE(before.frames[i].odom_pose.isApprox(after.frames[i].odom_pose));
  EXPECT_THROW(backend.append(scene(),pose(0),0),std::invalid_argument);
}
TEST_F(LoopTest, ValidatedLoopOptimizesGraphAndDenseExportWithoutChangingOdom) {
  auto config=Backend::Config{}; config.minimum_loop_seconds=20;
  Backend backend(root,config,false);
  // Return to the same scene after a circuit with 0.8 m accumulated odometry error.
  const std::vector<Pose,Eigen::aligned_allocator<Pose>> actual{
    pose(0),pose(3),pose(3,3),pose(0,3),pose(0)};
  for (std::size_t i=0;i<actual.size();++i) {
    Pose odom=actual[i]; odom.translation().x()+=.2*i;
    backend.append(scene(actual[i]),odom,i*6);
  }
  backend.processPending();
  const auto s=backend.snapshot();
  ASSERT_TRUE(s.error.empty())<<s.error;
  ASSERT_EQ(s.processed_keyframes,5u);
  ASSERT_EQ(s.loop_count,1u);
  EXPECT_LT(s.keyframes.back().translation().norm(),.4);
  EXPECT_NEAR(s.frames.back().odom_pose.translation().x(),.8,1e-9);
  std::string error; ASSERT_TRUE(backend.save(root+"/map.pcd",&error))<<error;
  Backend::Cloud saved; ASSERT_EQ(pcl::io::loadPCDFile(root+"/map.pcd",saved),0);
  EXPECT_EQ(saved.size(),actual.size()*1800);
  EXPECT_LT(std::abs(saved[4*1800].x-saved[0].x),.4);
}
TEST_F(LoopTest, DescriptorSimilarityAloneCannotCreateALoop) {
  auto config=Backend::Config{}; config.minimum_loop_seconds=20;
  Backend backend(root,config,false);
  const auto cloud=scene();
  backend.append(cloud,pose(0),0);
  backend.append(cloud,pose(40),40); // Same descriptor but no ICP overlap at odometry seed.
  backend.processPending();
  const auto s=backend.snapshot(); ASSERT_TRUE(s.error.empty())<<s.error;
  EXPECT_EQ(s.loop_count,0u); EXPECT_NEAR(s.keyframes.back().translation().x(),40,1e-6);
}
TEST_F(LoopTest, LoopMatchesAfterTurningAndPreservesHeading) {
  auto config=Backend::Config{}; config.minimum_loop_seconds=20;
  Backend backend(root,config,false);
  const Pose returned=pose(0,0,M_PI/2);
  backend.append(scene(),pose(0),0);
  backend.append(scene(pose(3)),pose(3.1),8);
  backend.append(scene(pose(3,3)),pose(3.2,3),16);
  backend.append(scene(returned),pose(.4,0,M_PI/2+.1),24);
  backend.processPending();
  const auto s=backend.snapshot(); ASSERT_TRUE(s.error.empty())<<s.error;
  ASSERT_EQ(s.loop_count,1u);
  const Pose error=returned.inverse()*s.keyframes.back();
  EXPECT_LT(error.translation().norm(),.25);
  EXPECT_LT(Eigen::AngleAxisd(error.linear()).angle(),.1);
}
TEST_F(LoopTest, GroundRobotLoopRecoversLargeVerticalOdomDrift) {
  auto config=Backend::Config{}; config.minimum_loop_seconds=20;
  config.planar_revisit_initialization=true; config.huber_delta=5.0;
  config.vertical_sigma=.03;
  Backend backend(root,config,false);
  const std::vector<Pose,Eigen::aligned_allocator<Pose>> actual{
    pose(0),pose(3),pose(3,3),pose(0,3),pose(0)};
  for (std::size_t i=0;i<actual.size();++i) {
    const Pose odom=poseWithZ(actual[i].translation().x(),actual[i].translation().y(),.425*i);
    backend.append(scene(actual[i]),odom,i*6);
  }
  backend.processPending();
  const auto s=backend.snapshot(); ASSERT_TRUE(s.error.empty())<<s.error;
  ASSERT_EQ(s.loop_count,1u);
  EXPECT_LT(std::abs(s.keyframes.back().translation().z()),.05);
  EXPECT_NEAR(s.frames.back().odom_pose.translation().z(),1.7,1e-9);
}
TEST_F(LoopTest, SaveUsesOnePoseSnapshotWhileNewScansArrive) {
  Backend backend(root,Backend::Config{},false);
  backend.append(scene(),pose(0),0); backend.processPending();
  bool appended=false; std::string error;
  Backend::Snapshot saved_snapshot;
  ASSERT_TRUE(backend.save(root+"/map.pcd",&error,[&] {
    if (!appended) { appended=true; backend.append(scene(),pose(2),1); }
    return false;
  }, &saved_snapshot))<<error;
  ASSERT_EQ(saved_snapshot.frames.size(),1u);
  EXPECT_DOUBLE_EQ(saved_snapshot.frames.front().stamp,0);
  Backend::Cloud saved; ASSERT_EQ(pcl::io::loadPCDFile(root+"/map.pcd",saved),0);
  EXPECT_EQ(saved.size(),1800u); EXPECT_EQ(backend.snapshot().frames.size(),2u);
  EXPECT_FALSE(backend.discardSavedChunks());
}
TEST_F(LoopTest, ExportsKeyframePosesLocalScansAndDescriptors) {
  Backend backend(root,Backend::Config{},false);
  for (int i=0;i<8;++i) backend.append(scene(pose(i*.2)),pose(i*.2),i*.2);
  backend.processPending();
  Backend::Snapshot saved; std::string error;
  ASSERT_TRUE(backend.save(root+"/map.pcd",&error,{},&saved))<<error;
  ASSERT_EQ(saved.keyframes.size(),2u);
  const auto directory=root+"/map.pcd.keyframes";
  ASSERT_TRUE(backend.exportKeyframes(directory,saved,&error))<<error;
  // Re-export replaces the directory and leaves no staging copies behind.
  ASSERT_TRUE(backend.exportKeyframes(directory,saved,&error))<<error;
  for (const auto &entry:std::filesystem::directory_iterator(root))
    EXPECT_EQ(entry.path().filename().string().find("map.pcd.keyframes."),std::string::npos);

  std::ifstream index(directory+"/keyframes.txt"); std::string line;
  std::getline(index,line); EXPECT_EQ(line,"# HENG_KEYFRAMES_1"); std::getline(index,line);
  for (std::size_t k=0;k<saved.keyframes.size();++k) {
    std::size_t id,points; double stamp,x,y,z,qx,qy,qz,qw;
    ASSERT_TRUE(index>>id>>stamp>>x>>y>>z>>qx>>qy>>qz>>qw>>points);
    EXPECT_EQ(id,k); EXPECT_DOUBLE_EQ(stamp,k*1.0);
    EXPECT_NEAR(x,saved.keyframes[k].translation().x(),1e-9);
    EXPECT_NEAR(std::abs(qw),1.0,1e-6);
    char name[32]; std::snprintf(name,sizeof(name),"/scans/%06zu.pcd",k);
    Backend::Cloud cloud;
    ASSERT_EQ(pcl::io::loadPCDFile(directory+name,cloud),0);
    ASSERT_EQ(cloud.size(),points); ASSERT_GT(points,0u);
    // Keyframe-local scans placed by the keyframe pose land back in the scene.
    for (const auto &p:cloud) {
      const auto world=saved.keyframes[k]*Eigen::Vector3d(p.x,p.y,p.z);
      EXPECT_LE(world.head<2>().lpNorm<Eigen::Infinity>(),8.2);
      EXPECT_GE(world.z(),-1.2); EXPECT_LE(world.z(),3.2);
    }
  }
  std::ifstream descriptors(directory+"/scan_context.bin",std::ios::binary);
  char magic[8]; std::uint32_t shape[2]; std::uint64_t count=0;
  descriptors.read(magic,8); descriptors.read(reinterpret_cast<char *>(shape),sizeof(shape));
  descriptors.read(reinterpret_cast<char *>(&count),sizeof(count));
  EXPECT_EQ(std::string(magic,8),"HENGSC01"); EXPECT_EQ(count,2u);
  EXPECT_EQ(std::filesystem::file_size(directory+"/scan_context.bin"),
            24u+count*shape[0]*shape[1]*sizeof(double));
}
}
