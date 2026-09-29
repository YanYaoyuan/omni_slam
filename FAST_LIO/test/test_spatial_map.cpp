#include "fast_lio/spatial_map.hpp"
#include "fast_lio/extent_safe_voxel_filter.hpp"
#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

namespace {
using Map=fast_lio::SpatialMap;
namespace fs=std::filesystem;
class SpatialMapTest: public ::testing::Test {
protected:
  std::string root;
  void SetUp() override { char p[]="/tmp/heng-spatial-map-XXXXXX"; root=::mkdtemp(p); }
  void TearDown() override { fs::remove_all(root); }
  static Map::Cloud cloud() {
    Map::Cloud result;
    for(int region=-2;region<=2;++region) for(int i=0;i<400;++i) {
      pcl::PointXYZINormal p{};
      p.x=region*100+i%20*.03; p.y=(i/20)*.03; p.z=(i%3)*.02;
      p.intensity=i+region*400; p.normal_x=1; p.curvature=.25;
      result.push_back(p);
    }
    auto invalid=result[0]; invalid.x=std::numeric_limits<float>::quiet_NaN();
    result.push_back(invalid); result.is_dense=false;
    return result;
  }
  static std::set<std::tuple<float,float,float,float>> records(const Map::Cloud &cloud) {
    std::set<std::tuple<float,float,float,float>> result;
    for(const auto &p:cloud) result.emplace(p.x,p.y,p.z,p.intensity);
    return result;
  }
};
TEST_F(SpatialMapTest, StreamsAllLegacyEncodingsAndMatchesExistingVoxelFilter) {
  const auto input=cloud(); Map::Cloud expected;
  fast_lio::pointcloud::extentSafeVoxelFilter(input,.3,&expected);
  for (const auto &encoding:{"binary","compressed","ascii"}) {
    const auto file=root+"/"+encoding+".pcd";
    if (std::string(encoding)=="binary") ASSERT_EQ(pcl::io::savePCDFileBinary(file,input),0);
    else if (std::string(encoding)=="compressed") ASSERT_EQ(pcl::io::savePCDFileBinaryCompressed(file,input),0);
    else ASSERT_EQ(pcl::io::savePCDFileASCII(file,input),0);
    const auto before=fs::file_size(file); const auto timestamp=fs::last_write_time(file);
    Map map(file,root+"/cache",Map::Config{});
    EXPECT_EQ(map.stats().input_points,input.size());
    EXPECT_EQ(map.stats().valid_points,input.size()-1);
    EXPECT_EQ(records(map.region(Eigen::Vector3d::Zero(),300)),records(expected));
    EXPECT_EQ(fs::file_size(file),before); EXPECT_EQ(fs::last_write_time(file),timestamp);
    EXPECT_EQ(map.preview().size(),expected.size());
    Map reused(file,root+"/cache",Map::Config{});
    EXPECT_TRUE(reused.stats().reused_index);
    EXPECT_EQ(records(reused.region(Eigen::Vector3d::Zero(),300)),records(expected));
  }
}
TEST_F(SpatialMapTest, LoadsOnlyNearbyTilesAndBoundsLruWithoutDanglingClouds) {
  auto config=Map::Config{}; config.cache_points=1; config.preview_points=3;
  const auto file=root+"/map.pcd"; ASSERT_EQ(pcl::io::savePCDFileBinary(file,cloud()),0);
  Map map(file,root+"/cache",config);
  auto nearby=map.region(Eigen::Vector3d::Zero(),10);
  ASSERT_FALSE(nearby.empty());
  for(const auto &p:nearby) EXPECT_LT(std::abs(p.x),10);
  EXPECT_LE(map.stats().cached_points,config.cache_points);
  EXPECT_EQ(map.preview().size(),3u);
  const auto first=map.tile(map.tilesFor(Eigen::Vector3d::Zero(),10).front());
  const float remembered=first->front().x;
  EXPECT_FALSE(map.region(Eigen::Vector3d(200,0,0),10).empty());
  EXPECT_EQ(first->front().x,remembered);
  EXPECT_LE(map.stats().cached_points,config.cache_points);
}
TEST_F(SpatialMapTest, ChangedPcdGetsANewIndexAndOversizedRegionFailsClearly) {
  const auto file=root+"/map.pcd"; auto input=cloud();
  ASSERT_EQ(pcl::io::savePCDFileBinary(file,input),0);
  auto config=Map::Config{}; config.region_points=1;
  Map first(file,root+"/cache",config);
  EXPECT_THROW(first.region(Eigen::Vector3d::Zero(),300),std::runtime_error);
  input.erase(input.begin(),input.begin()+500);
  ASSERT_EQ(pcl::io::savePCDFileBinary(file,input),0);
  Map second(file,root+"/cache",config);
  EXPECT_NE(first.directory(),second.directory());
  EXPECT_FALSE(second.stats().reused_index);
  EXPECT_EQ(second.stats().input_points,input.size());
}
TEST_F(SpatialMapTest, RejectsMissingEmptyAndNonfiniteMapsBeforePublishingIndex) {
  const auto file=root+"/map.pcd";
  EXPECT_THROW(Map(file,root+"/cache",Map::Config{}),std::exception);
  for (const auto &data:{std::string{},std::string{"nan nan nan\n"}}) {
    std::ofstream out(file);
    const int count=data.empty()?0:1;
    out<<"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "
       <<count<<"\nHEIGHT 1\nPOINTS "<<count<<"\nDATA ascii\n"<<data;
    out.close();
    EXPECT_THROW(Map(file,root+"/cache",Map::Config{}),std::runtime_error);
  }
  if (fs::exists(root+"/cache"))
    for (const auto &entry:fs::recursive_directory_iterator(root+"/cache"))
      EXPECT_NE(entry.path().filename(),"index.txt");
}
TEST_F(SpatialMapTest, CancellationDoesNotPublishPartialIndexOrChangePcd) {
  const auto file=root+"/map.pcd"; ASSERT_EQ(pcl::io::savePCDFileBinary(file,cloud()),0);
  const auto size=fs::file_size(file);
  EXPECT_THROW(Map(file,root+"/cache",Map::Config{},[]{return true;}),std::runtime_error);
  EXPECT_EQ(fs::file_size(file),size);
  for(const auto &p:fs::directory_iterator(root+"/cache")) EXPECT_FALSE(p.is_directory());
  EXPECT_NO_THROW(Map(file,root+"/cache",Map::Config{}));
}
TEST_F(SpatialMapTest, RejectsTruncatedBinaryAndMalformedCompressedPcd) {
  const auto file=root+"/bad.pcd"; ASSERT_EQ(pcl::io::savePCDFileBinary(file,cloud()),0);
  fs::resize_file(file,fs::file_size(file)-40);
  EXPECT_THROW(Map(file,root+"/cache",Map::Config{}),std::runtime_error);
  ASSERT_EQ(pcl::io::savePCDFileBinaryCompressed(file,cloud()),0);
  fs::resize_file(file,fs::file_size(file)-40);
  EXPECT_THROW(Map(file,root+"/cache",Map::Config{}),std::runtime_error);
}
TEST_F(SpatialMapTest, ConcurrentReadersReuseOneCommittedIndex) {
  const auto file=root+"/map.pcd"; ASSERT_EQ(pcl::io::savePCDFileBinary(file,cloud()),0);
  std::string dirs[2]; bool reused[2]{};
  auto read=[&](int i) {
    Map map(file,root+"/cache",Map::Config{});
    dirs[i]=map.directory(); reused[i]=map.stats().reused_index;
    EXPECT_FALSE(map.region(Eigen::Vector3d::Zero(),10).empty());
  };
  std::thread a(read,0),b(read,1); a.join(); b.join();
  EXPECT_EQ(dirs[0],dirs[1]); EXPECT_NE(reused[0],reused[1]);
}
}
