#include <gtest/gtest.h>
#include "preprocess.h"
#include "scan_time_window.hpp"
#include "patch_score.hpp"
#include "IMU_Processing.h"
#include "visual_time_alignment.hpp"
#include <cstring>
#include <limits>

namespace {
auto cloud()
{
  auto m=std::make_shared<sensor_msgs::msg::PointCloud2>();
  m->header.stamp.sec=100;
  m->height=2; m->width=2; m->point_step=26; m->row_step=56;
  using F=sensor_msgs::msg::PointField;
  const std::vector<std::tuple<std::string,uint32_t,uint8_t>> fields={
    {"x",0,F::FLOAT32},{"y",4,F::FLOAT32},{"z",8,F::FLOAT32},
    {"intensity",12,F::FLOAT32},{"ring",16,F::UINT16},{"timestamp",18,F::FLOAT64}};
  for (const auto &[name,offset,type]:fields) { F f; f.name=name;f.offset=offset;f.datatype=type;f.count=1;m->fields.push_back(f); }
  m->data.resize(m->height*m->row_step,0xAA);
  for (int i=0;i<4;++i) {
    auto *p=m->data.data()+(i/2)*m->row_step+(i%2)*m->point_step;
    float xyz[4]={float(i+1),0,0,7};uint16_t ring=i==3?16:1;double t=100.0+(3-i)*0.05;
    std::memcpy(p,xyz,16);std::memcpy(p+16,&ring,2);std::memcpy(p+18,&t,8);
  }
  return m;
}
Preprocess processor() { Preprocess p;p.lidar_type=VANJEE;p.N_SCANS=16;p.blind_sqr=0.01;return p; }
}
TEST(Vanjee, HandlesPackedFieldsPaddingEpochTimesAndOneBasedLastRing)
{
  auto p=processor();PointCloudXYZI::Ptr out(new PointCloudXYZI());
  p.process(cloud(),out);
  ASSERT_EQ(out->size(),4u);
  EXPECT_FLOAT_EQ(out->front().x,4.0f);
  EXPECT_NEAR(out->front().curvature,0,1e-4);
  EXPECT_NEAR(out->back().curvature,150,1e-3);
}
TEST(Vanjee, RejectsMissingTimestampRatherThanInventingTimes)
{
  auto p=processor();auto m=cloud();m->fields.pop_back();PointCloudXYZI::Ptr out(new PointCloudXYZI());
  EXPECT_THROW(p.process(m,out),std::invalid_argument);
}
TEST(Vanjee, RejectsTruncatedPayloadAndWrongDatatype)
{
  auto p=processor();auto m=cloud();PointCloudXYZI::Ptr out(new PointCloudXYZI());
  m->data.pop_back();EXPECT_THROW(p.process(m,out),std::invalid_argument);
  m=cloud();m->fields.back().datatype=sensor_msgs::msg::PointField::FLOAT32;
  EXPECT_THROW(p.process(m,out),std::invalid_argument);
}
TEST(Vanjee, FiltersNonfiniteCoordinatesAndUnusableTimestamps)
{
  auto p=processor();auto m=cloud();PointCloudXYZI::Ptr out(new PointCloudXYZI());
  float nan=std::numeric_limits<float>::quiet_NaN();std::memcpy(m->data.data(),&nan,4);
  double invalid=99;std::memcpy(m->data.data()+m->point_step+18,&invalid,8);
  p.process(m,out);EXPECT_EQ(out->size(),2u);
}
TEST(Ros2Time, PreservesSensorTimeAndRejectsInvalidTime)
{
  builtin_interfaces::msg::Time t;t.sec=100;t.nanosec=123456789;
  EXPECT_NEAR(stamp_seconds(t),100.123456789,1e-12);
  auto restored=stamp_from_seconds(stamp_seconds(t));EXPECT_EQ(restored.sec,t.sec);EXPECT_EQ(restored.nanosec,t.nanosec);
  EXPECT_THROW(stamp_from_seconds(-1),std::invalid_argument);
}

TEST(ScanTimeWindow, DoesNotConsumeFuturePointsWhenCameraIsFasterThanLidar)
{
  PointCloudXYZI input,current,future;
  for(float time:{0.0f,20.0f,40.0f,100.0f,180.0f}) {
    PointType p{};p.curvature=time;input.push_back(p);
  }
  fast_livo::split_remaining_scan(input,33.0,current,future);
  ASSERT_EQ(current.size(),2u);ASSERT_EQ(future.size(),3u);
  EXPECT_FLOAT_EQ(current.back().curvature,20.0f);
  EXPECT_FLOAT_EQ(future.front().curvature,7.0f);
  EXPECT_FLOAT_EQ(future.back().curvature,147.0f);
}

TEST(ScanTimeWindow, RepeatedCutsConservePointsAndTheirAbsoluteTimes)
{
  PointCloudXYZI pending,current,future;
  for(int i=0;i<200;++i) {PointType p{};p.x=i;p.curvature=i;pending.push_back(p);}
  std::vector<bool> seen(200,false);double previous=0;
  for(double end:{33.0,66.0,100.0,133.0,166.0,200.0}) {
    fast_livo::split_remaining_scan(pending,end-previous,current,future);
    for(const auto &p:current) {
      const auto id=static_cast<size_t>(p.x);
      EXPECT_FALSE(seen.at(id));seen[id]=true;
      EXPECT_GE(p.curvature,0);EXPECT_LT(p.curvature,end-previous);
      EXPECT_NEAR(previous+p.curvature,id,1e-5);
    }
    pending.swap(future);previous=end;
  }
  EXPECT_TRUE(pending.empty());
  EXPECT_TRUE(std::all_of(seen.begin(),seen.end(),[](bool value){return value;}));
}

TEST(ScanTimeWindow, BoundaryPointBelongsToNextUpdateAndInvalidTimeIsRejected)
{
  PointCloudXYZI input,current,future;PointType p{};p.curvature=50;input.push_back(p);
  fast_livo::split_remaining_scan(input,50,current,future);
  EXPECT_TRUE(current.empty());ASSERT_EQ(future.size(),1u);EXPECT_FLOAT_EQ(future[0].curvature,0);
  EXPECT_THROW(fast_livo::split_remaining_scan(input,0,current,future),std::invalid_argument);
}

TEST(ScanTimeWindow, ClampsSubMicrosecondRoundoffButRejectsNegativeAcquisitionTime)
{
  PointCloudXYZI input,current,future;PointType p{};
  p.curvature=-0.00001f;input.push_back(p);
  fast_livo::split_remaining_scan(input,33,current,future);
  ASSERT_EQ(current.size(),1u);EXPECT_FLOAT_EQ(current[0].curvature,0.0f);
  input[0].curvature=-0.01f;
  EXPECT_THROW(fast_livo::split_remaining_scan(input,33,current,future),std::invalid_argument);
}

TEST(PatchScore, CorrelationIsFiniteForConstantAndRepeatedReferencePatches)
{
  float a[]={1,2,3,4},b[]={11,12,13,14},constant[]={5,5,5,5};
  EXPECT_NEAR(fast_livo::patch_ncc(a,b,4),1.0,1e-12);
  EXPECT_DOUBLE_EQ(fast_livo::patch_ncc(a,constant,4),0.0);
  for(int i=0;i<100;++i) EXPECT_NEAR(fast_livo::patch_ncc(a,b,4),1.0,1e-12);
}

TEST(PatchScore, CoarseSearchLevelMustFitGradientAndBilinearNeighbors)
{
  EXPECT_TRUE(fast_livo::patch_in_frame(50,50,640,360,4,4));
  EXPECT_FALSE(fast_livo::patch_in_frame(50,50,640,360,4,16));
  EXPECT_TRUE(fast_livo::patch_in_frame(100,100,640,360,4,16));
  EXPECT_FALSE(fast_livo::patch_in_frame(320,-1,640,360,4,1));
  EXPECT_FALSE(fast_livo::patch_in_frame(std::numeric_limits<double>::quiet_NaN(),180,640,360,4,1));
}

TEST(ImuInitialization, EpochTimeAndFirstPointAreIntegratedExactlyOnce)
{
  ImuProcess imu;imu.set_imu_init_frame_num(1);
  LidarMeasureGroup group;StatesGroup state;group.lio_vio_flg=LIO;
  const double epoch=1779377443.0;
  auto measurement=[&](double start,double end) {
    MeasureGroup m;m.lio_time=epoch+end;
    for(double t=start;t<end+1e-6;t+=0.005) {
      auto msg=std::make_shared<sensor_msgs::msg::Imu>();
      msg->header.stamp=stamp_from_seconds(epoch+t);msg->linear_acceleration.z=9.81;
      m.imu.push_back(msg);
    }
    return m;
  };
  PointCloudXYZI::Ptr output(new PointCloudXYZI());
  group.measures.push_back(measurement(0.005,0.1));
  imu.Process2(group,state,output);
  ASSERT_FALSE(imu.imu_need_init);EXPECT_DOUBLE_EQ(group.last_lio_update_time,epoch+0.1);
  EXPECT_TRUE(output->empty());state.vel_end=V3D(1,0,0);
  group.measures.clear();group.measures.push_back(measurement(0.105,0.2));
  PointType p{};p.x=1;p.curvature=50;group.pcl_proc_cur->push_back(p);
  p.x=2;p.curvature=100;group.pcl_proc_cur->push_back(p);
  imu.Process2(group,state,output);
  EXPECT_NEAR(state.pos_end.x(),0.1,1e-6);
  ASSERT_EQ(output->size(),2u);
  EXPECT_NEAR((*output)[0].x,0.95,1e-5);EXPECT_NEAR((*output)[1].x,2.0,1e-5);
}

TEST(VisualTimeAlignment, IntegratesActualImageToLidarRotationWithBiasRemoved)
{
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> data;
  for(int i=0;i<5;++i) {
    auto m=std::make_shared<sensor_msgs::msg::Imu>();m->header.stamp=stamp_from_seconds(100+i*.01);
    m->angular_velocity.z=1.1;data.push_back(m);
  }
  auto rotation=fast_livo::rotation_between(data,100.005,100.035,V3D(0,0,.1));
  EXPECT_NEAR(Eigen::AngleAxisd(rotation).angle(),.03,1e-10);
  EXPECT_TRUE(fast_livo::rotation_between(data,100.02,100.02,V3D::Zero()).isApprox(M3D::Identity()));
  EXPECT_THROW(fast_livo::rotation_between(data,99,100,V3D::Zero()),std::invalid_argument);
}

TEST(ImuUnits, MovingInitializationDoesNotRecalibrateSiAccelerationScale)
{
  ImuProcess imu;imu.set_imu_init_frame_num(1);imu.set_acceleration_scale(1.0);
  LidarMeasureGroup group;StatesGroup state;group.lio_vio_flg=LIO;
  PointCloudXYZI::Ptr output(new PointCloudXYZI());
  auto measurement=[](double begin,double end,double acceleration) {
    MeasureGroup m;m.lio_time=end;
    for(double t=begin;t<end+1e-8;t+=.005) {
      auto x=std::make_shared<sensor_msgs::msg::Imu>();x->header.stamp=stamp_from_seconds(t);
      x->linear_acceleration.z=acceleration;m.imu.push_back(x);
    }
    return m;
  };
  auto init=measurement(100.005,100.1,10.4);
  // The last initialization sample is stationary; all subsequent samples are
  // stationary too. The earlier startup acceleration must not rescale them.
  std::const_pointer_cast<sensor_msgs::msg::Imu>(init.imu.back())->linear_acceleration.z=9.81;
  group.measures.push_back(init);imu.Process2(group,state,output);
  ASSERT_FALSE(imu.imu_need_init);EXPECT_GT(imu.IMU_mean_acc_norm,10.3);
  group.measures.clear();group.measures.push_back(measurement(100.105,100.2,9.81));
  PointType p{};p.x=2;p.curvature=50;group.pcl_proc_cur->push_back(p);
  imu.Process2(group,state,output);
  EXPECT_NEAR(state.vel_end.z(),0,1e-10);EXPECT_NEAR(state.pos_end.z(),0,1e-10);
  EXPECT_DOUBLE_EQ(imu.acceleration_scale(),1.0);
}
