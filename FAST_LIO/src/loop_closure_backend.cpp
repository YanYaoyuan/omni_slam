#include "fast_lio/loop_closure_backend.hpp"
#include "fast_lio/atomic_file.hpp"
#include "fast_lio/extent_safe_voxel_filter.hpp"
#include "fast_lio/trajectory_exclusion.hpp"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fast_lio {
namespace {
constexpr int kRings=20, kSectors=60;
constexpr double kRadius=80, kPi=3.14159265358979323846;
using Descriptor = Eigen::Matrix<double, kRings, kSectors>;
using Cloud = LoopClosureBackend::Cloud;
using Pose = LoopClosureBackend::Pose;
gtsam::Pose3 graphPose(const Pose &p) { return gtsam::Pose3(p.matrix()); }
Pose eigenPose(const gtsam::Pose3 &p) { Pose result; result.matrix()=p.matrix(); return result; }
gtsam::ISAM2Params graphParameters() {
  gtsam::ISAM2Params p;
  p.relinearizeSkip=1;
  p.relinearizeThreshold=0.01;
  return p;
}

Descriptor describe(const Cloud &cloud) {
  Descriptor result=Descriptor::Zero();
  for (const auto &p: cloud) {
    const double radius=std::hypot(p.x,p.y);
    if (!std::isfinite(radius) || !std::isfinite(p.z) || radius<=0.1 || radius>=kRadius) continue;
    const int ring=std::min(kRings-1, int(radius/kRadius*kRings));
    const double angle=std::atan2(p.y,p.x)+kPi;
    const int sector=std::min(kSectors-1, int(angle/(2*kPi)*kSectors));
    result(ring,sector)=std::max(result(ring,sector), std::max(0.1,double(p.z)+2.0));
  }
  return result;
}
std::pair<double,int> descriptorDistance(const Descriptor &current, const Descriptor &candidate) {
  double best=1; int best_shift=0;
  for (int shift=0; shift<kSectors; ++shift) {
    double similarity=0; int compared=0;
    for (int col=0; col<kSectors; ++col) {
      const auto a=current.col(col), b=candidate.col((col+shift)%kSectors);
      const double norms=a.norm()*b.norm();
      if (norms>1e-9) { similarity+=a.dot(b)/norms; ++compared; }
    }
    if (compared<6) continue;
    const double distance=1-similarity/compared;
    if (distance<best) { best=distance; best_shift=shift; }
  }
  return {best,best_shift};
}
Cloud::Ptr downsample(const Cloud &cloud, double leaf) {
  pcl::VoxelGrid<pcl::PointXYZINormal> filter;
  filter.setInputCloud(cloud.makeShared());
  filter.setLeafSize(leaf,leaf,leaf);
  Cloud::Ptr result(new Cloud); filter.filter(*result); return result;
}
bool writeBinaryPcd(const std::string &path, const Cloud &cloud) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity normal_x normal_y normal_z curvature\n"
    "SIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\nWIDTH " << cloud.size()
    << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << cloud.size() << "\nDATA binary\n";
  for (const auto &p: cloud) {
    const float fields[8]={p.x,p.y,p.z,p.intensity,p.normal_x,p.normal_y,p.normal_z,p.curvature};
    output.write(reinterpret_cast<const char *>(fields),sizeof(fields));
  }
  output.flush();
  return static_cast<bool>(output);
}
// Removes a tree written by exportKeyframes: regular files and plain subdirectories.
void removeTree(const std::string &path) {
  DIR *directory=::opendir(path.c_str());
  if (!directory) { ::unlink(path.c_str()); return; }
  while (const dirent *entry=::readdir(directory)) {
    const std::string name=entry->d_name;
    if (name=="." || name=="..") continue;
    const std::string child=path+"/"+name;
    struct stat info{};
    if (::lstat(child.c_str(),&info)==0 && S_ISDIR(info.st_mode)) removeTree(child);
    else ::unlink(child.c_str());
  }
  ::closedir(directory);
  ::rmdir(path.c_str());
}
}

struct LoopClosureBackend::Impl {
  struct Keyframe {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::size_t frame;
    double stamp;
    Pose odom;
  };
  using Keyframes = std::vector<Keyframe,Eigen::aligned_allocator<Keyframe>>;
  Config config;
  MapArchive archive;
  mutable std::mutex state_mutex;
  std::mutex processing_mutex, wait_mutex;
  std::condition_variable wake;
  bool stopping{false};
  std::thread worker;
  Keyframes keys;
  MapArchive::Poses optimized;
  std::size_t processed{0}, loops{0};
  double last_stamp{-std::numeric_limits<double>::infinity()}, cycle_ms{0};
  std::string error;
  bool graph_failed{false};
  std::vector<Descriptor,Eigen::aligned_allocator<Descriptor>> descriptors;
  gtsam::ISAM2 graph;

  Impl(const std::string &root, Config c): config(c), archive(root,c.chunk_bytes,c.preview_points),
    graph(graphParameters()) {
    archive.setTrajectoryExclusion(c.trajectory_exclusion_radius,c.lidar_in_imu);
    if (!(c.frequency_hz>0 && c.frequency_hz<=1 && c.keyframe_distance>0 && c.keyframe_angle>0 &&
          c.keyframe_seconds>0 && c.voxel_size>0 && c.candidates>0 && c.candidates<=5 &&
          c.maximum_correspondence>0 && c.maximum_fitness>0 && c.minimum_overlap>0 &&
          c.minimum_overlap<=1 && c.overlap_distance>0 && c.minimum_loop_seconds>=0 &&
          c.descriptor_threshold>0 && c.descriptor_threshold<1 && c.huber_delta>0 &&
          c.vertical_sigma>0 &&
          std::isfinite(c.keyframe_export_voxel) && c.keyframe_export_voxel>=0 &&
          std::isfinite(c.export_frame_interval) && c.export_frame_interval>=0))
      throw std::invalid_argument("invalid background loop configuration");
  }
  ~Impl() {
    { std::lock_guard<std::mutex> lock(wait_mutex); stopping=true; }
    wake.notify_all(); if (worker.joinable()) worker.join();
  }
  bool align(const Cloud::Ptr &source, const Cloud::Ptr &target, const Pose &initial,
             Pose *relative) const {
    if (source->size()<30 || target->size()<30) return false;
    pcl::IterativeClosestPoint<pcl::PointXYZINormal,pcl::PointXYZINormal> icp;
    icp.setInputSource(source); icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(config.maximum_correspondence);
    icp.setMaximumIterations(40); icp.setTransformationEpsilon(1e-6);
    Cloud aligned;
    icp.align(aligned,initial.matrix().cast<float>());
    if (!icp.hasConverged() || !icp.getFinalTransformation().allFinite() ||
        icp.getFitnessScore(config.maximum_correspondence)>config.maximum_fitness) return false;
    pcl::KdTreeFLANN<pcl::PointXYZINormal> tree; tree.setInputCloud(target);
    std::vector<int> indices(1); std::vector<float> distances(1);
    std::size_t inliers=0;
    for (const auto &p: aligned)
      if (tree.nearestKSearch(p,1,indices,distances)>0 &&
          distances[0]<=config.overlap_distance*config.overlap_distance) ++inliers;
    if (double(inliers)/aligned.size()<config.minimum_overlap) return false;
    relative->matrix()=icp.getFinalTransformation().cast<double>();
    return true;
  }
  void process() {
    std::lock_guard<std::mutex> serial(processing_mutex);
    if (!config.enabled || graph_failed) return;
    const auto start=std::chrono::steady_clock::now();
    Keyframes pending;
    { std::lock_guard<std::mutex> lock(state_mutex); pending=keys; }
    try {
      for (; processed<pending.size(); ++processed) {
        const auto i=processed;
        const auto source=downsample(archive.readFrame(pending[i].frame),config.voxel_size);
        const auto descriptor=describe(*source);
        gtsam::NonlinearFactorGraph factors;
        gtsam::Values initial;
        if (!i) {
          initial.insert(i,graphPose(pending[i].odom));
          factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            i,graphPose(pending[i].odom),gtsam::noiseModel::Isotropic::Sigma(6,1e-6));
        } else {
          const auto delta=graphPose(pending[i-1].odom.inverse()*pending[i].odom);
          initial.insert(i,graph.calculateEstimate<gtsam::Pose3>(i-1).compose(delta));
          gtsam::Vector6 sigmas; sigmas<<.02,.02,.02,.10,.10,.10;
          factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            i-1,i,delta,gtsam::noiseModel::Diagonal::Sigmas(sigmas));
        }
        bool closed=false;
        if (config.enabled) {
          // Ring keys shortlist before the cyclic Scan Context comparison.
          std::vector<std::pair<double,std::size_t>> candidates;
          const auto ring_key=descriptor.rowwise().mean().eval();
          for (std::size_t j=0; j<descriptors.size(); ++j) {
            if (pending[i].stamp-pending[j].stamp<config.minimum_loop_seconds) continue;
            candidates.emplace_back((ring_key-descriptors[j].rowwise().mean()).squaredNorm(),j);
          }
          const auto n=std::min(config.candidates,candidates.size());
          std::partial_sort(candidates.begin(),candidates.begin()+n,candidates.end());
          for (std::size_t k=0; k<n && !closed; ++k) {
            const auto j=candidates[k].second;
            const auto match=descriptorDistance(descriptor,descriptors[j]);
            if (match.first>config.descriptor_threshold) continue;
            const auto target=downsample(archive.readFrame(pending[j].frame),config.voxel_size);
            Pose guess=pending[j].odom.inverse()*pending[i].odom;
            // A VBot re-observing the same place should have near-zero relative
            // height even when its long-loop odometry has accumulated Z drift.
            if (config.planar_revisit_initialization) guess.translation().z()=0;
            // Scan Context provides yaw; retain odometry roll/pitch and translation.
            const double old_yaw=std::atan2(guess.linear()(1,0),guess.linear()(0,0));
            guess.linear()=Eigen::AngleAxisd(match.second*2*kPi/kSectors-old_yaw,
              Eigen::Vector3d::UnitZ()).toRotationMatrix()*guess.linear();
            Pose measurement;
            if (!align(source,target,guess,&measurement)) continue;
            gtsam::Vector6 sigmas; sigmas<<.05,.05,.05,.10,.10,config.vertical_sigma;
            const auto noise=gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(config.huber_delta),
              gtsam::noiseModel::Diagonal::Sigmas(sigmas));
            factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(j,i,graphPose(measurement),noise);
            closed=true;
          }
        }
        graph.update(factors,initial);
        // A loop can require several relinearizations to propagate the correction.
        // Keep this bounded and exclusively on the background worker.
        if (closed) for (int pass=0; pass<3; ++pass) graph.update();
        descriptors.push_back(descriptor);
        MapArchive::Poses result;
        result.reserve(i+1);
        const auto estimates=graph.calculateEstimate();
        for (std::size_t j=0; j<=i; ++j) result.push_back(eigenPose(estimates.at<gtsam::Pose3>(j)));
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          optimized=std::move(result); if (closed) ++loops;
        }
      }
      std::lock_guard<std::mutex> lock(state_mutex);
      cycle_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      error.clear();
    } catch (const std::exception &e) {
      // An optimizer exception may leave partially updated factors. Preserve
      // the last complete correction rather than retrying a half-committed graph.
      graph_failed=true;
      std::lock_guard<std::mutex> lock(state_mutex); error=e.what();
    }
  }
};

LoopClosureBackend::LoopClosureBackend(const std::string &root,Config config,bool start_worker)
  : impl_(new Impl(root,config)) {
  if (start_worker && config.enabled) impl_->worker=std::thread([this] {
    const auto period=std::chrono::duration<double>(1/impl_->config.frequency_hz);
    std::unique_lock<std::mutex> lock(impl_->wait_mutex);
    while (!impl_->stopping) {
      if (impl_->wake.wait_for(lock,period,[this] { return impl_->stopping; })) break;
      lock.unlock(); impl_->process(); lock.lock();
    }
  });
}
LoopClosureBackend::~LoopClosureBackend()=default;
void LoopClosureBackend::append(const Cloud &scan,const Pose &pose,double stamp) {
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (!std::isfinite(stamp) || stamp<impl_->last_stamp || !pose.matrix().allFinite())
    throw std::invalid_argument("mapping frame timestamps must be monotonic and poses finite");
  bool keyframe=impl_->keys.empty();
  if (!keyframe) {
    const auto &last=impl_->keys.back();
    const Pose delta=last.odom.inverse()*pose;
    keyframe=delta.translation().norm()>=impl_->config.keyframe_distance ||
      Eigen::AngleAxisd(delta.linear()).angle()>=impl_->config.keyframe_angle ||
      stamp-last.stamp>=impl_->config.keyframe_seconds;
  }
  const auto id=impl_->keys.size()-(keyframe?0:1);
  const Pose relative=keyframe?Pose::Identity():impl_->keys.back().odom.inverse()*pose;
  const auto frame=impl_->archive.append(scan,pose,stamp,id,relative);
  if (keyframe) impl_->keys.push_back({frame,stamp,pose});
  impl_->last_stamp=stamp;
}
LoopClosureBackend::Snapshot LoopClosureBackend::snapshot() const {
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
  Snapshot result; result.frames=impl_->archive.frames(); result.keyframes=impl_->optimized;
  result.processed_keyframes=result.keyframes.size(); result.loop_count=impl_->loops;
  result.last_cycle_ms=impl_->cycle_ms; result.error=impl_->error;
  // Pending odometry is extended with the last completed graph correction.
  Pose correction=Pose::Identity();
  if (!result.keyframes.empty()) {
    const auto last=result.keyframes.size()-1;
    correction=result.keyframes.back()*impl_->keys[last].odom.inverse();
  }
  for (std::size_t i=result.keyframes.size(); i<impl_->keys.size(); ++i)
    result.keyframes.push_back(correction*impl_->keys[i].odom);
  return result;
}
LoopClosureBackend::Cloud LoopClosureBackend::preview(const Snapshot &s) const {
  return impl_->archive.preview(s.keyframes,s.frames.size());
}
bool LoopClosureBackend::save(const std::string &path,std::string *error,const std::function<bool()> &cancel, Snapshot *saved_snapshot) {
  const auto s=snapshot();
  const bool saved=impl_->archive.exportPcd(
    path,s.keyframes,error,cancel,s.frames.size(),impl_->config.export_frame_interval);
  if (saved && saved_snapshot) *saved_snapshot=s;
  return saved;
}
bool LoopClosureBackend::exportKeyframes(const std::string &directory,const Snapshot &s,
                                         std::string *error,const std::function<bool()> &cancel) const {
  const auto fail=[error](const std::string &message) { if (error) *error=message; return false; };
  if (directory.empty() || directory.back()=='/') return fail("keyframe export directory is invalid");
  if (s.frames.empty() || s.keyframes.empty()) return fail("no keyframes to export");
  Impl::Keyframes keys;
  { std::lock_guard<std::mutex> lock(impl_->state_mutex); keys=impl_->keys; }
  if (keys.size()<s.keyframes.size()) return fail("keyframe snapshot does not belong to this session");
  const auto &config=impl_->config;
  const std::string stage=directory+".tmp."+std::to_string(static_cast<long long>(::getpid()))+"."+
    std::to_string(static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::string message;
  try {
    if (::mkdir(stage.c_str(),0750)!=0 || ::mkdir((stage+"/scans").c_str(),0750)!=0)
      throw std::runtime_error("cannot create keyframe staging directory: "+stage);
    // Same swept volume as the exported PCD, so keyframes rebuild the same map.
    TrajectoryExclusion exclusion(config.trajectory_exclusion_radius);
    for (const auto &frame: s.frames)
      exclusion.add(s.keyframes.at(frame.keyframe)*frame.keyframe_from_frame*config.lidar_in_imu);
    std::ofstream index(stage+"/keyframes.txt");
    index << "# HENG_KEYFRAMES_1\n# index stamp x y z qx qy qz qw points\n" << std::setprecision(17);
    // scan_context.bin: "HENGSC01", uint32 rings, uint32 sectors, uint64 count,
    // then count row-major rings x sectors float64 matrices (host byte order).
    std::ofstream descriptors(stage+"/scan_context.bin",std::ios::binary);
    const char magic[8]={'H','E','N','G','S','C','0','1'};
    const std::uint32_t shape[2]={kRings,kSectors};
    const std::uint64_t count=s.keyframes.size();
    descriptors.write(magic,sizeof(magic));
    descriptors.write(reinterpret_cast<const char *>(shape),sizeof(shape));
    descriptors.write(reinterpret_cast<const char *>(&count),sizeof(count));
    std::size_t next=0;
    for (std::size_t k=0; k<s.keyframes.size(); ++k) {
      if (cancel && cancel()) throw std::runtime_error("keyframe export canceled");
      Cloud scan;
      for (; next<s.frames.size() && s.frames[next].keyframe==k; ++next) {
        const auto &frame=s.frames[next];
        const Pose world=s.keyframes[k]*frame.keyframe_from_frame;
        for (const auto &p: impl_->archive.readFrame(next)) {
          const Eigen::Vector3d point(p.x,p.y,p.z);
          if (exclusion.contains(world*point)) continue;
          const Eigen::Vector3d local=frame.keyframe_from_frame*point;
          auto q=p; q.x=local.x(); q.y=local.y(); q.z=local.z();
          scan.push_back(q);
        }
      }
      if (config.keyframe_export_voxel>0 && !scan.empty())
        pointcloud::extentSafeVoxelFilter(scan,config.keyframe_export_voxel,&scan);
      char name[32]; std::snprintf(name,sizeof(name),"%06zu.pcd",k);
      const std::string file=stage+"/scans/"+name;
      if (!scan.empty() && (!writeBinaryPcd(file,scan) || !sync_file(file,&message)))
        throw std::runtime_error("keyframe scan write failed: "+file+" "+message);
      const auto &pose=s.keyframes[k];
      const Eigen::Quaterniond q(pose.linear());
      index << k << ' ' << keys[k].stamp << ' ' << pose.translation().x() << ' '
            << pose.translation().y() << ' ' << pose.translation().z() << ' ' << q.x() << ' '
            << q.y() << ' ' << q.z() << ' ' << q.w() << ' ' << scan.size() << '\n';
      // Identical to the loop detector's descriptor of this keyframe.
      const auto descriptor=describe(*downsample(impl_->archive.readFrame(keys[k].frame),config.voxel_size));
      for (int r=0; r<kRings; ++r) for (int c=0; c<kSectors; ++c) {
        const double value=descriptor(r,c);
        descriptors.write(reinterpret_cast<const char *>(&value),sizeof(value));
      }
    }
    if (next!=s.frames.size()) throw std::runtime_error("archived frames reference unknown keyframes");
    index.close(); descriptors.close();
    if (!index || !descriptors) throw std::runtime_error("keyframe index write failed");
    if (!sync_file(stage+"/keyframes.txt",&message) || !sync_file(stage+"/scan_context.bin",&message) ||
        !sync_directory(stage+"/scans",&message) || !sync_directory(stage,&message))
      throw std::runtime_error(message);
    const std::string previous=stage+".old";
    struct stat info{};
    const bool replace=::lstat(directory.c_str(),&info)==0;
    if (replace && ::rename(directory.c_str(),previous.c_str())!=0)
      throw std::runtime_error("cannot move previous keyframe export aside: "+directory);
    if (::rename(stage.c_str(),directory.c_str())!=0) {
      if (replace) ::rename(previous.c_str(),directory.c_str());
      throw std::runtime_error("cannot publish keyframe export: "+directory);
    }
    if (replace) removeTree(previous);
    if (!sync_directory(parent_directory(directory),&message)) return fail(message);
  } catch (const std::exception &e) {
    removeTree(stage);
    return fail(e.what());
  }
  return true;
}
void LoopClosureBackend::processPending() { impl_->process(); }
std::string LoopClosureBackend::directory() const { return impl_->archive.directory(); }
bool LoopClosureBackend::discardSavedChunks() { return impl_->archive.discardSavedChunks(); }
}  // namespace fast_lio
