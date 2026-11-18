#pragma once

// Optional read-only observation of the offline estimator, for synchronized
// image / tracked feature / incremental point-cloud experiment recordings.
#include <cstdlib>
#include <filesystem>
#include <fstream>

class ExperimentCapture
{
public:
  ExperimentCapture()
  {
    const char *directory=std::getenv("FAST_LIVO_CAPTURE_DIR");
    if (!directory || !*directory) return;
    root_=directory;
    std::filesystem::create_directories(root_);
    index_.open(root_/"frames.csv");
    if (!index_) throw std::runtime_error("cannot open capture index");
    index_<<"id,stamp,image_stamp,x,y,z,features,points\n";
    index_<<std::setprecision(17);
  }

  void observe(const LIVMapper &mapper)
  {
    if (!index_.is_open() || mapper.p_imu->imu_need_init ||
        mapper.LidarMeasures.lio_vio_flg!=VIO ||
        mapper.LidarMeasures.last_lio_update_time<=last_stamp_) return;
    const auto &vio=*mapper.vio_manager;
    if (vio.img_rgb.empty() || !vio.new_frame_) return;
    const double stamp=mapper.LidarMeasures.last_lio_update_time;
    const auto prefix=(root_/std::to_string(count_)).string();
    if (!cv::imwrite(prefix+".jpg",vio.img_rgb,{cv::IMWRITE_JPEG_QUALITY,94}))
      throw std::runtime_error("capture image write failed");
    std::ofstream features(prefix+".csv");
    features<<"u,v,error_reduced\n";
    const auto &sub=*vio.visual_submap;
    for (std::size_t i=0;i<sub.voxel_points.size();++i) {
      const V2D pixel=vio.new_frame_->w2c(sub.voxel_points[i]->pos_);
      if (!pixel.allFinite()) continue;
      features<<pixel.x()<<','<<pixel.y()<<','
        <<(sub.errors[i]<=sub.propa_errors[i] ? 1 : 0)<<'\n';
    }
    std::ofstream cloud(prefix+".xyz",std::ios::binary);
    for (const auto &point:*mapper.feats_down_body) {
      const V3D body=mapper.extR*V3D(point.x,point.y,point.z)+mapper.extT;
      const V3D world=mapper._state.rot_end*body+mapper._state.pos_end;
      const float xyz[3]={float(world.x()),float(world.y()),float(world.z())};
      cloud.write(reinterpret_cast<const char *>(xyz),sizeof(xyz));
    }
    if (!features || !cloud) throw std::runtime_error("capture geometry write failed");
    index_<<count_<<','<<stamp<<','<<mapper.LidarMeasures.measures.back().image_time<<','
      <<mapper._state.pos_end.x()<<','<<mapper._state.pos_end.y()<<','<<mapper._state.pos_end.z()<<','
      <<sub.voxel_points.size()<<','<<mapper.feats_down_body->size()<<'\n';
    if (!index_) throw std::runtime_error("capture index write failed");
    last_stamp_=stamp;++count_;
  }

private:
  std::filesystem::path root_;
  std::ofstream index_;
  double last_stamp_=-1;
  std::size_t count_=0;
};
