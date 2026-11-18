/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include "visual_time_alignment.hpp"
#include "scan_time_window.hpp"
#include <filesystem>
#include <pcl/common/transforms.h>

LIVMapper::LIVMapper(const rclcpp::Node::SharedPtr &node)
    : extT(0, 0, 0), extR(M3D::Identity()), node_(node)
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  auto &nh = *node_;
  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = runtime_directory();
  initializeFiles();
  initializeComponents();
  path.header.stamp = node_->now();
  path.header.frame_id = odom_frame_;
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(rclcpp::Node &nh)
{
  read_parameter<string>(nh, "common/lid_topic", lid_topic, "/livox/lidar");
  read_parameter<string>(nh, "common/imu_topic", imu_topic, "/livox/imu");
  read_parameter<bool>(nh, "common/ros_driver_bug_fix", ros_driver_fix_en, false);
  read_parameter<int>(nh, "common/img_en", img_en, 1);
  read_parameter<int>(nh, "common/lidar_en", lidar_en, 1);
  read_parameter<string>(nh, "common/img_topic", img_topic, "/left_camera/image");

  read_parameter<bool>(nh, "vio/normal_en", normal_en, true);
  read_parameter<double>(nh, "vio/lidar_bootstrap_s", lidar_bootstrap_s_, 0.0);
  read_parameter<bool>(nh, "vio/lidar_timed_fusion", lidar_timed_fusion_, false);
  read_parameter<bool>(nh, "runtime/keyframe_archive", keyframe_archive_en_, false);
  if (keyframe_archive_en_ && img_en && !lidar_timed_fusion_)
    throw std::invalid_argument("keyframe archive with images requires lidar-timed fusion");
  read_parameter<double>(nh, "vio/max_image_age_s", max_image_age_s_, 0.04);
  if (!(max_image_age_s_>0 && max_image_age_s_<=0.1))
    throw std::invalid_argument("vio.max_image_age_s must be in (0,0.1]");
  if (!std::isfinite(lidar_bootstrap_s_) || lidar_bootstrap_s_<0)
    throw std::invalid_argument("vio.lidar_bootstrap_s must be nonnegative and finite");
  read_parameter<bool>(nh, "vio/inverse_composition_en", inverse_composition_en, false);
  read_parameter<int>(nh, "vio/max_iterations", max_iterations, 5);
  read_parameter<double>(nh, "vio/img_point_cov", IMG_POINT_COV, 100);
  read_parameter<bool>(nh, "vio/raycast_en", raycast_en, false);
  read_parameter<bool>(nh, "vio/exposure_estimate_en", exposure_estimate_en, true);
  read_parameter<double>(nh, "vio/inv_expo_cov", inv_expo_cov, 0.2);
  read_parameter<int>(nh, "vio/grid_size", grid_size, 5);
  read_parameter<int>(nh, "vio/grid_n_height", grid_n_height, 17);
  read_parameter<int>(nh, "vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  read_parameter<int>(nh, "vio/patch_size", patch_size, 8);
  read_parameter<double>(nh, "vio/outlier_threshold", outlier_threshold, 1000);

  read_parameter<double>(nh, "time_offset/exposure_time_init", exposure_time_init, 0.0);
  read_parameter<double>(nh, "time_offset/img_time_offset", img_time_offset, 0.0);
  read_parameter<double>(nh, "time_offset/imu_time_offset", imu_time_offset, 0.0);
  read_parameter<double>(nh, "time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  read_parameter<bool>(nh, "uav/imu_rate_odom", imu_prop_enable, false);
  read_parameter<bool>(nh, "uav/gravity_align_en", gravity_align_en, false);

  read_parameter<string>(nh, "evo/seq_name", seq_name, "01");
  read_parameter<bool>(nh, "evo/pose_output_en", pose_output_en, false);
  read_parameter<double>(nh, "imu/gyr_cov", gyr_cov, 1.0);
  read_parameter<double>(nh, "imu/acc_cov", acc_cov, 1.0);
  read_parameter<int>(nh, "imu/imu_int_frame", imu_int_frame, 3);
  read_parameter<bool>(nh, "imu/imu_en", imu_en, false);
  read_parameter<bool>(nh, "imu/gravity_est_en", gravity_est_en, true);
  read_parameter<bool>(nh, "imu/ba_bg_est_en", ba_bg_est_en, true);

  read_parameter<double>(nh, "preprocess/blind", p_pre->blind, 0.01);
  read_parameter<double>(nh, "preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  read_parameter<bool>(nh, "preprocess/hilti_en", hilti_en, false);
  read_parameter<int>(nh, "preprocess/lidar_type", p_pre->lidar_type, AVIA);
  read_parameter<int>(nh, "preprocess/scan_line", p_pre->N_SCANS, 6);
  read_parameter<int>(nh, "preprocess/point_filter_num", p_pre->point_filter_num, 3);
  read_parameter<bool>(nh, "preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  read_parameter<int>(nh, "pcd_save/interval", pcd_save_interval, -1);
  read_parameter<bool>(nh, "pcd_save/pcd_save_en", pcd_save_en, false);
  read_parameter<int>(nh, "pcd_save/type", pcd_save_type, 0);
  read_parameter<bool>(nh, "image_save/img_save_en", img_save_en, false);
  read_parameter<int>(nh, "image_save/interval", img_save_interval, 1);

  read_parameter<bool>(nh, "pcd_save/colmap_output_en", colmap_output_en, false);
  read_parameter<double>(nh, "pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  read_parameter<vector<double>>(nh, "extrin_calib/extrinsic_T", extrinT, vector<double>{0.0, 0.0, 0.0});
  read_parameter<vector<double>>(nh, "extrin_calib/extrinsic_R", extrinR, vector<double>{1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0});
  read_parameter<vector<double>>(nh, "extrin_calib/Pcl", cameraextrinT, vector<double>{0.0, 0.0, 0.0});
  read_parameter<vector<double>>(nh, "extrin_calib/Rcl", cameraextrinR, vector<double>{1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0});
  read_parameter<double>(nh, "debug/plot_time", plot_time, -10);
  read_parameter<int>(nh, "debug/frame_cnt", frame_cnt, 6);

  read_parameter<double>(nh, "publish/blind_rgb_points", blind_rgb_points, 0.01);
  read_parameter<int>(nh, "publish/pub_scan_num", pub_scan_num, 1);
  read_parameter<bool>(nh, "publish/pub_effect_point_en", pub_effect_point_en, false);
  read_parameter<bool>(nh, "publish/dense_map_en", dense_map_en, false);

  read_parameter(nh, "common.odom_frame", odom_frame_, std::string("omni_odom"));
  read_parameter(nh, "common.tracking_frame", tracking_frame_, std::string("omni_imu_link"));
  read_parameter(nh, "common.camera_frame", camera_frame_, std::string("omni_rgb_camera_optical_frame"));
  read_parameter(nh, "publish.tf_en", publish_tf_, false);
  read_parameter(nh, "runtime.map_file_path", managed_map_path_, std::string(""));
  if (!managed_map_path_.empty() && (!keyframe_archive_en_ ||
      !std::filesystem::path(managed_map_path_).is_absolute()))
    throw std::invalid_argument("managed map save requires keyframe archive and an absolute output path");
  read_parameter(nh, "runtime.output_directory", runtime_directory(), std::string("/tmp/fast_livo"));
  if (runtime_directory().empty()) throw std::invalid_argument("runtime.output_directory is empty");
  if (runtime_directory().back() != '/') runtime_directory() += '/';
  for (const auto &part : {"Log", "Log/result", "Log/pcd", "Log/image", "Log/Colmap/sparse/0", "Log/Colmap/images", "Log/ref_cur_combine"})
    std::filesystem::create_directories(runtime_directory() + part);
  if (extrinT.size()!=3 || extrinR.size()!=9 || cameraextrinT.size()!=3 || cameraextrinR.size()!=9)
    throw std::invalid_argument("extrinsic translations require 3 values and rotations require 9");
  if (p_pre->N_SCANS<1 || p_pre->N_SCANS>128 || p_pre->point_filter_num<1 || grid_n_height<1 || patch_size<2 || patch_pyrimid_level<1 || pub_scan_num<1)
    throw std::invalid_argument("invalid scan, patch, grid or publication configuration");
  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  const int width = node_->declare_parameter<int>("camera.width", 640);
  const int height = node_->declare_parameter<int>("camera.height", 480);
  const double scale = node_->declare_parameter<double>("camera.scale", 1.0);
  const double fx = node_->declare_parameter<double>("camera.fx", 0.0);
  const double fy = node_->declare_parameter<double>("camera.fy", 0.0);
  const double cx = node_->declare_parameter<double>("camera.cx", 0.0);
  const double cy = node_->declare_parameter<double>("camera.cy", 0.0);
  const auto d = node_->declare_parameter<std::vector<double>>("camera.distortion", {0.0,0.0,0.0,0.0,0.0});
  if (width<1 || height<1 || !(scale>0 && scale<=1) || (img_en && (!(fx>0) || !(fy>0))) || d.size()!=5)
    throw std::invalid_argument("camera requires positive intrinsics, dimensions, scale in (0,1] and five distortion coefficients; rectify rational models upstream");
  vio_manager->cam = new vk::PinholeCamera(width,height,scale,fx>0?fx:1.0,fy>0?fy:1.0,cx,cy,d[0],d[1],d[2],d[3],d[4]);

  vio_manager->update_enabled = node_->declare_parameter<bool>("vio.update_enabled",true);
  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();
  nominal_Rci_=vio_manager->Rci;
  nominal_Pci_=vio_manager->Pci;

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);
  const double acceleration_scale=node_->declare_parameter<double>("imu.acceleration_scale",0.0);
  p_imu->set_acceleration_scale(acceleration_scale);
  p_imu->max_interpolation_gap_s = node_->declare_parameter<double>("imu.max_interpolation_gap_s", 0.0);
  if (!std::isfinite(p_imu->max_interpolation_gap_s) || p_imu->max_interpolation_gap_s < 0)
    throw std::invalid_argument("imu.max_interpolation_gap_s must be finite and nonnegative");

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  trajectory_file_.open(root_dir+"trajectory.txt");
  timing_file_.open(root_dir+"timing.csv");
  timing_file_ << "stamp,update_type,processing_ms,points,visual_points,position_correction_m,rotation_correction_rad,image_age_ms\n";
  if(colmap_output_en) fout_points.open(runtime_directory() + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_en) fout_lidar_pos.open(runtime_directory() + "Log/pcd/lidar_poses.txt", std::ios::out);
  if(img_save_en) fout_visual_pos.open(runtime_directory() + "Log/image/image_poses.txt", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers()
{
  const auto sensors = rclcpp::SensorDataQoS().keep_last(1000);
#ifdef FAST_LIVO_USE_LIVOX
  if (p_pre->lidar_type==AVIA)
    sub_livox_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, sensors, std::bind(&LIVMapper::livox_pcl_cbk,this,std::placeholders::_1));
  else
#endif
  {
    if (p_pre->lidar_type==AVIA) throw std::invalid_argument("Livox CustomMsg requires -DFAST_LIVO_USE_LIVOX=ON; use PointCloud2 for VBot");
    sub_pcl = node_->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, sensors, std::bind(&LIVMapper::standard_pcl_cbk,this,std::placeholders::_1));
  }
  sub_imu = node_->create_subscription<sensor_msgs::msg::Imu>(imu_topic, sensors, std::bind(&LIVMapper::imu_cbk,this,std::placeholders::_1));
  if (img_en) sub_img = node_->create_subscription<sensor_msgs::msg::Image>(img_topic, rclcpp::SensorDataQoS().keep_last(20), std::bind(&LIVMapper::img_cbk,this,std::placeholders::_1));
  pubLaserCloudFullRes = node_->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", 10);
  pubSubVisualMap = node_->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_visual_sub_map_before", 10);
  pubLaserCloudEffect = node_->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_effected", 10);
  pubLaserCloudMap = node_->create_publisher<sensor_msgs::msg::PointCloud2>("Laser_map", 10);
  pubLaserCloudDyn = node_->create_publisher<sensor_msgs::msg::PointCloud2>("dyn_obj", 10);
  pubLaserCloudDynRmed = node_->create_publisher<sensor_msgs::msg::PointCloud2>("dyn_obj_removed", 10);
  pubLaserCloudDynDbg = node_->create_publisher<sensor_msgs::msg::PointCloud2>("dyn_obj_dbg_hist", 10);
  pubImuPropOdom = node_->create_publisher<nav_msgs::msg::Odometry>("imu_propagate", 10);
  pubOdomAftMapped = node_->create_publisher<nav_msgs::msg::Odometry>("Odometry", 10);
  pubPath = node_->create_publisher<nav_msgs::msg::Path>("path", 10);
  plane_pub = node_->create_publisher<visualization_msgs::msg::Marker>("planner_normal", 10);
  voxel_pub = node_->create_publisher<visualization_msgs::msg::MarkerArray>("voxels", 10);
  pubNormal = node_->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker", 10);
  mavros_pose_publisher = node_->create_publisher<geometry_msgs::msg::PoseStamped>("mavros_pose", 10);
  pubImage = node_->create_publisher<sensor_msgs::msg::Image>("rgb_img", 2);
  pubKeyframe = node_->create_publisher<omni_slam_interfaces::msg::LivoKeyframe>("keyframes", 10);
  // A slow inference process must not accumulate an input queue or reliable DDS backpressure.
  pubSemanticFrame = node_->create_publisher<omni_slam_interfaces::msg::LivoSemanticFrame>(
      "semantic_frames", rclcpp::QoS(1).best_effort());
  if (!managed_map_path_.empty())
    map_save_service_ = node_->create_service<std_srvs::srv::Trigger>("map_save",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) { saveManagedMap(*response); });
  voxelmap_manager->voxel_map_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("planes", 1);
  if (publish_tf_) tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
  if (imu_prop_enable) imu_prop_timer = node_->create_wall_timer(std::chrono::milliseconds(4), std::bind(&LIVMapper::imu_prop_callback,this));
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  if (!(lidar_timed_fusion_ && LidarMeasures.lio_vio_flg==VIO))
    p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  const size_t input_points = LidarMeasures.lio_vio_flg==VIO ? pcl_w_wait_pub->size() : feats_undistort->size();
  if (input_points==0) return;
  const auto began=std::chrono::steady_clock::now();
  const auto prior_position=_state.pos_end;
  const auto prior_rotation=_state.rot_end;
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
  if (!_state.pos_end.allFinite() || !_state.rot_end.allFinite() || !_state.cov.allFinite())
    throw std::runtime_error("nonfinite estimator state");
  if (keyframe_archive_en_ && !p_imu->imu_need_init &&
      (LidarMeasures.lio_vio_flg==VIO ||
       (LidarMeasures.lio_vio_flg==LIO && (slam_mode_!=LIVO || !camera_update_pending_))))
    archiveKeyframe();
  const double milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
  if (!p_imu->imu_need_init && LidarMeasures.last_lio_update_time>0) {
    const Eigen::Quaterniond q(_state.rot_end);
    trajectory_file_ << std::setprecision(17) << LidarMeasures.last_lio_update_time << ' '
      << _state.pos_end.transpose() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << ' '
      << static_cast<int>(LidarMeasures.lio_vio_flg) << '\n';
    timing_file_ << std::setprecision(17) << LidarMeasures.last_lio_update_time << ','
      << static_cast<int>(LidarMeasures.lio_vio_flg) << ',' << milliseconds << ',' << input_points << ',' << (LidarMeasures.lio_vio_flg==VIO ? vio_manager->total_points : 0) << ','
      << (_state.pos_end-prior_position).norm() << ',' << Eigen::AngleAxisd(prior_rotation.transpose()*_state.rot_end).angle() << ','
      << (lidar_timed_fusion_ && LidarMeasures.lio_vio_flg==VIO ?
          (LidarMeasures.measures.back().lio_time-LidarMeasures.measures.back().image_time)*1000.0 : 0.0) << '\n';
  }
}

void LIVMapper::archiveKeyframe()
{
  const double stamp=LidarMeasures.last_lio_update_time;
  const double angle=Eigen::AngleAxisd(keyframe_archive_last_rotation_.transpose()*_state.rot_end).angle();
  if (keyframe_archive_last_time_>=0 && stamp-keyframe_archive_last_time_<keyframe_archive_seconds_ &&
      (_state.pos_end-keyframe_archive_last_position_).norm()<1.0 && angle<0.2617993878) return;
  const auto directory=std::filesystem::path(root_dir)/"keyframes";
  if (!keyframe_archive_index_.is_open()) {
    std::filesystem::create_directories(directory);
    keyframe_archive_index_.open(root_dir+"keyframes.csv");
    keyframe_archive_index_<<"stamp,id,x,y,z,qx,qy,qz,qw,gx,gy,gz,vx,vy,vz,ax_world,ay_world,az_world,gyro_norm,points\n";
  }
  PointCloudXYZRGB cloud;cloud.reserve(feats_down_body->size());
  const bool color=LidarMeasures.lio_vio_flg==VIO && !vio_manager->img_rgb.empty();
  for (const auto &p:*feats_down_body) {
    const V3D body=extR*V3D(p.x,p.y,p.z)+extT;
    PointTypeRGB out{};out.x=body.x();out.y=body.y();out.z=body.z();out.r=out.g=out.b=128;
    if (color) {
      const V3D camera=vio_manager->new_frame_->w2f(_state.rot_end*body+_state.pos_end);
      if (camera.z()>0 && camera.allFinite()) {
        const V2D pixel=vio_manager->cam->world2cam(camera);
        if (pixel.allFinite() && vio_manager->cam->isInFrame(pixel.cast<int>(),3)) {
          const auto rgb=vio_manager->getInterpolatedPixel(vio_manager->img_rgb,pixel);
          out.r=rgb.z();out.g=rgb.y();out.b=rgb.x();
        }
      }
    }
    cloud.push_back(out);
  }
  const auto file=directory/(std::to_string(keyframe_archive_count_)+".pcd");
  if (pcl::io::savePCDFileBinary(file.string(),cloud)<0) throw std::runtime_error("keyframe PCD write failed");
  const auto q=Eigen::Quaterniond(_state.rot_end);
  V3D acceleration=V3D::Zero();double gyro=std::numeric_limits<double>::quiet_NaN();
  if (!LidarMeasures.measures.back().imu.empty()) {
    const auto &m=*LidarMeasures.measures.back().imu.back();
    acceleration=_state.rot_end*(V3D(m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z)*
        p_imu->acceleration_scale()-_state.bias_a);
    gyro=(V3D(m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z)-_state.bias_g).norm();
  }
  auto &out=keyframe_archive_index_;out<<std::setprecision(17)<<stamp<<','<<keyframe_archive_count_;
  for (double v:{_state.pos_end.x(),_state.pos_end.y(),_state.pos_end.z(),q.x(),q.y(),q.z(),q.w(),
       _state.gravity.x(),_state.gravity.y(),_state.gravity.z(),_state.vel_end.x(),_state.vel_end.y(),_state.vel_end.z(),
       acceleration.x(),acceleration.y(),acceleration.z(),gyro}) out<<','<<v;
  out<<','<<cloud.size()<<'\n';out.flush();
  if (!out) throw std::runtime_error("keyframe index write failed");
  if (!managed_map_path_.empty()) {
    Eigen::Matrix4f transform=Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0)=_state.rot_end.cast<float>();
    transform.block<3,1>(0,3)=_state.pos_end.cast<float>();
    managed_keyframes_.emplace_back(file.string(), transform);
  }
  const bool semantic_requested=color && pubSemanticFrame && pubSemanticFrame->get_subscription_count()>0;
  if ((pubKeyframe && pubKeyframe->get_subscription_count()>0) || semantic_requested) {
    omni_slam_interfaces::msg::LivoKeyframe message;
    message.header.stamp=stamp_from_seconds(stamp);message.header.frame_id=odom_frame_;
    message.id=keyframe_archive_count_;message.tracking_frame=tracking_frame_;
    set_posestamp(message.pose);
    pcl::toROSMsg(cloud,message.cloud);message.cloud.header=message.header;
    message.cloud.header.frame_id=tracking_frame_;
    message.visual_update=color;
    if (color) message.image_stamp=stamp_from_seconds(LidarMeasures.measures.back().image_time);
    pubKeyframe->publish(message);
    if (semantic_requested) {
      omni_slam_interfaces::msg::LivoSemanticFrame packet;
      packet.keyframe=message;
      cv_bridge::CvImage image;
      image.header.stamp=message.image_stamp;image.header.frame_id=camera_frame_;
      image.encoding=sensor_msgs::image_encodings::BGR8;
      // Consumers always receive pinhole pixels, also for a distorted input camera.
      // pinhole_cam is only initialized by the optional upstream COLMAP path.
      // initializeComponents always creates cam as a PinholeCamera in this port.
      static_cast<vk::PinholeCamera*>(vio_manager->cam)->undistortImage(vio_manager->img_rgb,image.image);
      packet.image=*image.toImageMsg();
      auto &info=packet.camera_info;
      info.header=image.header;info.width=image.image.cols;info.height=image.image.rows;
      info.distortion_model="plumb_bob";info.d.assign(5,0.0);
      info.k={vio_manager->cam->fx(),0.0,vio_manager->cam->cx(),
              0.0,vio_manager->cam->fy(),vio_manager->cam->cy(),0.0,0.0,1.0};
      info.r={1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};
      info.p={info.k[0],0.0,info.k[2],0.0,0.0,info.k[4],info.k[5],0.0,0.0,0.0,1.0,0.0};
      const auto &cw=vio_manager->new_frame_->T_f_w_;
      const M3D rotation=cw.rotation_matrix()*_state.rot_end;
      const V3D translation=cw.rotation_matrix()*_state.pos_end+cw.translation();
      const Eigen::Quaterniond cq(rotation);
      auto &tf=packet.camera_from_tracking;
      tf.translation.x=translation.x();tf.translation.y=translation.y();tf.translation.z=translation.z();
      tf.rotation.x=cq.x();tf.rotation.y=cq.y();tf.rotation.z=cq.z();tf.rotation.w=cq.w();
      pubSemanticFrame->publish(packet);
    }
  }
  keyframe_archive_last_time_=stamp;keyframe_archive_last_position_=_state.pos_end;
  keyframe_archive_last_rotation_=_state.rot_end;++keyframe_archive_count_;
}

void LIVMapper::saveManagedMap(std_srvs::srv::Trigger::Response &response)
{
  // The mapper spins callbacks on its estimator thread, so this is a coherent
  // snapshot of completed keyframes. Atomic rename preserves the previous map.
  try {
    if (managed_keyframes_.empty()) throw std::runtime_error("no completed keyframes");
    PointCloudXYZRGB::Ptr map(new PointCloudXYZRGB());
    for (const auto &entry:managed_keyframes_) {
      PointCloudXYZRGB local,world;
      if (pcl::io::loadPCDFile(entry.first,local)<0) throw std::runtime_error("cannot read keyframe");
      pcl::transformPointCloud(local,world,entry.second);*map+=world;
    }
    pcl::VoxelGrid<PointTypeRGB> filter;filter.setLeafSize(0.15f,0.15f,0.15f);
    filter.setInputCloud(map);PointCloudXYZRGB filtered;filter.filter(filtered);
    if (filtered.empty()) throw std::runtime_error("empty map");
    const std::filesystem::path destination(managed_map_path_);
    std::filesystem::create_directories(destination.parent_path());
    const auto temporary=managed_map_path_+".tmp.pcd";
    if (pcl::io::savePCDFileBinary(temporary,filtered)<0) throw std::runtime_error("map write failed");
    std::filesystem::rename(temporary,destination);
    response.success=true;response.message="saved frontend odom-frame keyframe map: "+managed_map_path_;
  } catch (const std::exception &error) {
    response.success=false;response.message=error.what();
  }
}

void LIVMapper::handleVIO() 
{
  if (lidar_timed_fusion_) {
    const auto &measurement=LidarMeasures.measures.back();
    const double age=measurement.lio_time-measurement.image_time;
    const M3D relative=fast_livo::rotation_between(measurement.imu,
        measurement.image_time,measurement.lio_time,_state.bias_g);
    // Evaluate the image at its acquisition pose while correcting the state at
    // the LiDAR epoch. Translation is a short-interval constant-velocity model.
    vio_manager->Rci=nominal_Rci_*relative;
    vio_manager->Pci=nominal_Pci_+vio_manager->Rci*_state.rot_end.transpose()*_state.vel_end*age;
    vio_manager->Jdphi_dR=vio_manager->Rci;
    const V3D center=-vio_manager->Rci.transpose()*vio_manager->Pci;
    M3D cross;cross<<SKEW_SYM_MATRX(center);
    vio_manager->Jdp_dR=-vio_manager->Rci*cross;
  }
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  geoQuat = quaternion_from_rpy(RotMtoEuler(_state.rot_end)(0),RotMtoEuler(_state.rot_end)(1),RotMtoEuler(_state.rot_end)(2));
  publish_odometry(pubOdomAftMapped);
  publish_path(pubPath);
  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO() 
{    
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) 
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(runtime_directory() + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "open fail\n");
    } 
    else 
    {
      evoFile.open(runtime_directory() + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = quaternion_from_rpy(euler_cur(0), euler_cur(1), euler_cur(2));
  if (slam_mode_ != LIVO || (lidar_timed_fusion_ && !camera_update_pending_)) publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  if (slam_mode_ != LIVO || (lidar_timed_fusion_ && !camera_update_pending_)) { publish_path(pubPath); publish_mavros(mavros_pose_publisher); }

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = runtime_directory() + "Log/pcd/all_raw_points.pcd";
    std::string downsampled_points_dir = runtime_directory() + "Log/pcd/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  rclcpp::WallRate rate(1000);
  while (rclcpp::ok()) 
  {
    rclcpp::spin_some(node_);
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  acc_avr = acc_avr * p_imu->acceleration_scale() - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback()
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && stamp_seconds(prop_imu_buffer.front().header.stamp) < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = stamp_seconds(prop_imu_buffer[i].header.stamp) - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = stamp_seconds(newest_imu.header.stamp) - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = odom_frame_;
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom->publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(extR * p_body_lidar + extT);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = stamp_seconds(msg->header.stamp) + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
    lid_header_time_buffer.clear();
  }
  // RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "get point cloud at time: %.6f", stamp_seconds(msg->header.stamp));
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  if (ptr->empty()) { mtx_buffer.unlock(); return; }
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

#ifdef FAST_LIVO_USE_LIVOX
void LIVMapper::livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver2::msg::CustomMsg::SharedPtr msg(new livox_ros_driver2::msg::CustomMsg(*msg_in));
  // if ((abs(stamp_seconds(msg->header.stamp) - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   RCLCPP_WARN(rclcpp::get_logger("fast_livo"), "lidar jumps %.3f\n", stamp_seconds(msg->header.stamp) - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = stamp_from_seconds(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - stamp_seconds(msg->header.stamp)) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - stamp_seconds(msg->header.stamp);
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = stamp_seconds(msg->header.stamp);
  RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
    lid_header_time_buffer.clear();
  }
  // RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "get point cloud at time: %.6f", stamp_seconds(msg->header.stamp));
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  if (ptr->empty()) { mtx_buffer.unlock(); return; }
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}
#endif


void LIVMapper::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  // RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "get imu at time: %.6f", stamp_seconds(msg_in->header.stamp));
  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
  msg->header.stamp = stamp_from_seconds(stamp_seconds(msg->header.stamp) - imu_time_offset);
  double timestamp = stamp_seconds(msg->header.stamp);

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    RCLCPP_WARN(rclcpp::get_logger("fast_livo"), "IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = stamp_from_seconds(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   RCLCPP_WARN(rclcpp::get_logger("fast_livo"), "imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr &msg_in)
{
  if (!img_en) return;
  sensor_msgs::msg::Image::SharedPtr msg(new sensor_msgs::msg::Image(*msg_in));
  // if ((abs(stamp_seconds(msg->header.stamp) - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   RCLCPP_WARN(rclcpp::get_logger("fast_livo"), "img jumps %.3f\n", stamp_seconds(msg->header.stamp) - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = stamp_from_seconds(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  // double msg_header_time =  stamp_seconds(msg->header.stamp);
  double msg_header_time = stamp_seconds(msg->header.stamp) + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    RCLCPP_WARN(rclcpp::get_logger("fast_livo"), "Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (slam_mode_==LIVO && lidar_timed_fusion_) {
    if (camera_update_pending_) {
      meas.measures.clear();meas.measures.push_back(pending_camera_measure_);
      meas.lio_vio_flg=VIO;camera_update_pending_=false;
      return true;
    }
    if (lid_raw_data_buffer.empty() || imu_buffer.empty()) return false;
    auto cloud=lid_raw_data_buffer.front();
    if (cloud->size()<2) throw std::runtime_error("empty LiDAR scan");
    const double begin=lid_header_time_buffer.front();
    const double end=begin+cloud->back().curvature*0.001;
    if (last_timestamp_imu<end) return false;
    MeasureGroup m;m.lio_time=end;
    while (!imu_buffer.empty() && stamp_seconds(imu_buffer.front()->header.stamp)<=end) {
      m.imu.push_back(imu_buffer.front());imu_buffer.pop_front();
    }
    while (!img_time_buffer.empty() && img_time_buffer.front()<=end) {
      m.img=img_buffer.front();m.image_time=img_time_buffer.front();
      img_time_buffer.pop_front();img_buffer.pop_front();
    }
    if (!m.img.empty() && end-m.image_time<=max_image_age_s_ && !m.imu.empty() &&
        stamp_seconds(m.imu.front()->header.stamp)<=m.image_time) {
      pending_camera_measure_=m;
      pending_camera_measure_.vio_time=end;
      camera_update_pending_=true;
    }
    if (meas.last_lio_update_time<0) meas.last_lio_update_time=begin;
    meas.lidar=cloud;meas.pcl_proc_cur=cloud;
    meas.lidar_frame_beg_time=begin;meas.lidar_frame_end_time=end;
    lid_raw_data_buffer.pop_front();lid_header_time_buffer.pop_front();
    meas.measures.clear();meas.measures.push_back(m);meas.lio_vio_flg=LIO;
    return true;
  }
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (imu_buffer.empty() && imu_en) return false;
  // Initialize all modes on complete LiDAR scans. Short image-timed slices
  // otherwise change both the IMU initialization window and first map geometry.
  const bool bootstrap = slam_mode_==LIVO && lidar_bootstrap_s_>0 && !visual_bootstrap_finished_ &&
    (p_imu->imu_need_init || !is_first_frame ||
     meas.last_lio_update_time-_first_lidar_time<lidar_bootstrap_s_);
  if (slam_mode_==LIVO && !bootstrap && !visual_bootstrap_finished_) {
    visual_bootstrap_finished_=true;
    meas.lio_vio_flg=WAIT;
    meas.measures.clear();
    pcl_w_wait_pub->clear();
  }
  if (slam_mode_==LIVO && !bootstrap) {
    while (!img_time_buffer.empty() && img_time_buffer.front()<=meas.last_lio_update_time) {
      img_time_buffer.pop_front();img_buffer.pop_front();
    }
  }
  if (img_buffer.empty() && img_en && !bootstrap) return false;

  switch (bootstrap ? ONLY_LIO : slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (stamp_seconds(imu_buffer.front()->header.stamp) > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.clear();
    meas.measures.push_back(m);
    // RCLCPP_DEBUG(rclcpp::get_logger("fast_livo"), "ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = stamp_seconds(imu_buffer.back()->header.stamp);

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (stamp_seconds(imu_buffer.front()->header.stamp) > m.lio_time) break;

        if (stamp_seconds(imu_buffer.front()->header.stamp) > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // stamp_seconds(imu_buffer.front()->header.stamp));
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      PointCloudXYZI remaining;
      remaining.swap(*meas.pcl_proc_next);
      fast_livo::split_remaining_scan(remaining,
          (m.lio_time-meas.last_lio_update_time)*1000.0,
          *meas.pcl_proc_cur,*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = stamp_seconds(imu_buffer.front()->header.stamp);

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = stamp_seconds(imu_buffer.front()->header.stamp);
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   stamp_seconds(imu_buffer.front()->header.stamp));
      // }
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  RCLCPP_ERROR(rclcpp::get_logger("fast_livo"), "out sync");
}

void LIVMapper::publish_img_rgb(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = stamp_from_seconds(lidar_timed_fusion_ && LidarMeasures.lio_vio_flg==VIO ?
      LidarMeasures.measures.back().image_time : LidarMeasures.last_lio_update_time);
  out_msg.header.frame_id = camera_frame_;
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage->publish(*out_msg.toImageMsg());
}

// Provide output format for LiDAR-visual BA
void LIVMapper::publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (lidar_timed_fusion_ && LidarMeasures.lio_vio_flg==LIO && camera_update_pending_) return;
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  static int pub_num = 1;
  pub_num++;

  if (LidarMeasures.lio_vio_flg == VIO)
  {
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num >= pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255; else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255; else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255; else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  if (slam_mode_ == LIVO && LidarMeasures.lio_vio_flg == VIO)
  {
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  if (LidarMeasures.lio_vio_flg != VIO)
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = stamp_from_seconds(LidarMeasures.last_lio_update_time); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = odom_frame_;
  pubLaserCloudFullRes->publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  double update_time = 0.0;
  if (LidarMeasures.lio_vio_flg == VIO) {
    update_time = LidarMeasures.measures.back().vio_time;
  } else { // LIO / LO
    update_time = LidarMeasures.measures.back().lio_time;
  }
  std::stringstream ss_time;
  ss_time << std::fixed << std::setprecision(6) << update_time;

  if (pcd_save_en)
  {
    static int scan_wait_num = 0;

    switch (pcd_save_type)
    {
      case 0: /** world frame **/
        if (slam_mode_ == LIVO)
        {
          *pcl_wait_save += *laserCloudWorldRGB;
        }
        else
        {
          *pcl_wait_save_intensity += *pcl_w_wait_pub;
        }
        if(LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO) scan_wait_num++;
        break;

      case 1: /** body frame **/
        if (LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO)
        {
          int size = feats_undistort->points.size();
          PointCloudXYZI::Ptr laserCloudBody(new PointCloudXYZI(size, 1));
          for (int i = 0; i < size; i++)
          {
            RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudBody->points[i]);
          }
          *pcl_wait_save_intensity += *laserCloudBody;
          scan_wait_num++;
          cout << "save body frame points: " << pcl_wait_save_intensity->points.size() << endl;
        }
        pcd_save_interval = 1;
        
        break;

      default:
        pcd_save_interval = 1;
        scan_wait_num++;
        break;
    }
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      string all_points_dir(string(runtime_directory() + "Log/pcd/") + ss_time.str() + string(".pcd"));

      pcl::PCDWriter pcd_writer;

      cout << "current scan saved to " << all_points_dir << endl;
      if (pcl_wait_save->points.size() > 0)
      {
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
        PointCloudXYZRGB().swap(*pcl_wait_save);
      }
      if(pcl_wait_save_intensity->points.size() > 0)
      {
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
        PointCloudXYZI().swap(*pcl_wait_save_intensity);
      }
      scan_wait_num = 0;
    }
    
    if(LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO)
    {
      Eigen::Quaterniond q(_state.rot_end);
      fout_lidar_pos << std::fixed << std::setprecision(6);
      fout_lidar_pos <<  LidarMeasures.measures.back().lio_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.x() << " " << q.y() << " " << q.z()
          << " " << q.w() << " " << endl;
    }
  }
  if (img_save_en && LidarMeasures.lio_vio_flg == VIO)
  {
    static int img_wait_num = 0;
    img_wait_num++;

    if (img_save_interval > 0 && img_wait_num >= img_save_interval)
    {
      imwrite(string(runtime_directory() + "Log/image/") + ss_time.str() + string(".png"), vio_manager->img_rgb);
      
      Eigen::Quaterniond q(_state.rot_end);
      fout_visual_pos << std::fixed << std::setprecision(6);
      fout_visual_pos << LidarMeasures.measures.back().vio_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
      img_wait_num = 0;
    }
  }

  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  if(LidarMeasures.lio_vio_flg == VIO)  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = stamp_from_seconds(LidarMeasures.last_lio_update_time);
    laserCloudmsg.header.frame_id = odom_frame_;
    pubSubVisualMap->publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = stamp_from_seconds(LidarMeasures.last_lio_update_time);
  laserCloudFullRes3.header.frame_id = odom_frame_;
  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = odom_frame_;
  odomAftMapped.child_frame_id = tracking_frame_;
  odomAftMapped.header.stamp = stamp_from_seconds(LidarMeasures.last_lio_update_time); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);
  const int pose_index[6]={3,4,5,0,1,2};
  for (int row=0;row<6;++row) for (int col=0;col<6;++col)
    odomAftMapped.pose.covariance[row*6+col]=_state.cov(pose_index[row],pose_index[col]);
  const auto velocity=_state.rot_end.transpose()*_state.vel_end;
  odomAftMapped.twist.twist.linear.x=velocity.x();
  odomAftMapped.twist.twist.linear.y=velocity.y();
  odomAftMapped.twist.twist.linear.z=velocity.z();


  if (tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header=odomAftMapped.header; tf.child_frame_id=tracking_frame_;
    tf.transform.translation.x=_state.pos_end.x();
    tf.transform.translation.y=_state.pos_end.y();
    tf.transform.translation.z=_state.pos_end.z();
    tf.transform.rotation=geoQuat;
    tf_broadcaster_->sendTransform(tf);
  }
  pubOdomAftMapped->publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = stamp_from_seconds(LidarMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = odom_frame_;
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher->publish(msg_body_pose);
}

void LIVMapper::publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = stamp_from_seconds(LidarMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = odom_frame_;
  path.poses.push_back(msg_body_pose);
  pubPath->publish(path);
}
