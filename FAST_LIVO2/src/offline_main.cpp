// Deterministic transport for experiments: original callbacks and estimator,
// synchronous draining, no wall-time playback or DDS sensor delivery.
#include "LIVMapper.h"
#include "experiment_capture.h"
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <sys/resource.h>
#include <array>
#include <cstring>

namespace {
void read_exact(char *data, std::size_t size)
{
  if (!std::cin.read(data, static_cast<std::streamsize>(size)))
    throw std::runtime_error("truncated experiment input; completion marker missing");
}

template<class T> std::shared_ptr<T> deserialize(const std::vector<uint8_t> &bytes)
{
  rclcpp::SerializedMessage serialized(bytes.size());
  auto &raw=serialized.get_rcl_serialized_message();
  std::memcpy(raw.buffer,bytes.data(),bytes.size());
  raw.buffer_length=bytes.size();
  auto message=std::make_shared<T>();
  rclcpp::Serialization<T>().deserialize_message(&serialized,message.get());
  return message;
}

std::string escaped(const std::string &value)
{
  std::string result;
  for (char c:value) {
    if(c=='"'||c=='\\') result+='\\';
    if(c=='\n') result+="\\n";
    else result+=c;
  }
  return result;
}
}

int main(int argc,char **argv)
{
  rclcpp::init(argc,argv);
  const auto start=std::chrono::steady_clock::now();
  std::unique_ptr<LIVMapper> mapper;
  std::array<uint64_t,3> counts{};
  std::size_t updates=0;
  bool completed=false;
  std::string error;
  int result=0;
  try {
    mapper=std::make_unique<LIVMapper>(std::make_shared<rclcpp::Node>("fast_livo_offline"));
    mapper->initializeSubscribersAndPublishers();
    ExperimentCapture capture;
    char magic[9];read_exact(magic,sizeof(magic));
    if(std::memcmp(magic,"OMNILIVO1",9)!=0) throw std::runtime_error("invalid offline input signature");
    while(rclcpp::ok()) {
      uint8_t type;uint32_t size;
      read_exact(reinterpret_cast<char*>(&type),1);
      read_exact(reinterpret_cast<char*>(&size),4);
      if(size>(256U<<20)) throw std::runtime_error("oversized offline event");
      std::vector<uint8_t> payload(size);
      read_exact(reinterpret_cast<char*>(payload.data()),size);
      if(type==0) {
        if(size!=24) throw std::runtime_error("invalid completion counts");
        std::array<uint64_t,3> expected;
        std::memcpy(expected.data(),payload.data(),24);
        if(counts!=expected) throw std::runtime_error("input count mismatch");
        completed=true;break;
      }
      if(type==1) {mapper->imu_cbk(deserialize<sensor_msgs::msg::Imu>(payload));++counts[0];}
      else if(type==2) {mapper->standard_pcl_cbk(deserialize<sensor_msgs::msg::PointCloud2>(payload));++counts[1];}
      else if(type==3) {
        if(size<16) throw std::runtime_error("truncated image metadata");
        int64_t stamp;uint32_t width,height;
        std::memcpy(&stamp,payload.data(),8);std::memcpy(&width,payload.data()+8,4);std::memcpy(&height,payload.data()+12,4);
        if(stamp<0 || !width || !height || 16+uint64_t(width)*height*3!=size)
          throw std::runtime_error("invalid image dimensions or timestamp");
        auto image=std::make_shared<sensor_msgs::msg::Image>();
        image->header.stamp.sec=stamp/1000000000LL;image->header.stamp.nanosec=stamp%1000000000LL;
        image->header.frame_id="omni_rgb_camera_optical_frame";
        image->width=width;image->height=height;image->encoding="bgr8";image->step=width*3;
        image->data.assign(payload.begin()+16,payload.end());
        mapper->img_cbk(image);++counts[2];
      } else throw std::runtime_error("unknown offline event type");
      while(mapper->sync_packages(mapper->LidarMeasures)) {
        mapper->handleFirstFrame();mapper->processImu();mapper->stateEstimationAndMapping();++updates;
        capture.observe(*mapper);
      }
      if(mapper->img_buffer.size()>1000 || mapper->lid_raw_data_buffer.size()>1000 || mapper->imu_buffer.size()>20000)
        throw std::runtime_error("sensor synchronization stalled; bounded input queue exceeded");
    }
    if(!completed) throw std::runtime_error("experiment interrupted before input completion");
    mapper->savePCD();
  } catch(const std::exception &e) {
    error=e.what();result=1;
    RCLCPP_ERROR(rclcpp::get_logger("fast_livo_offline"),"%s",e.what());
  }
  if(mapper) {
    mapper->trajectory_file_.flush();mapper->timing_file_.flush();
    struct rusage resource{};getrusage(RUSAGE_SELF,&resource);
    std::ofstream stats(mapper->root_dir+"offline_transport.json");
    stats<<std::setprecision(17)
      <<"{\"completed\":"<<(completed&&result==0?"true":"false")
      <<",\"exit_code\":"<<result<<",\"error\":\""<<escaped(error)<<"\""
      <<",\"input_counts\":{\"imu\":"<<counts[0]<<",\"lidar\":"<<counts[1]<<",\"image\":"<<counts[2]<<"}"
      <<",\"estimator_cycles\":"<<updates
      <<",\"pending_imu\":"<<mapper->imu_buffer.size()<<",\"pending_lidar\":"<<mapper->lid_raw_data_buffer.size()
      <<",\"pending_image\":"<<mapper->img_buffer.size()
      <<",\"wall_s\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
      <<",\"peak_rss_mib\":"<<resource.ru_maxrss/1024.0<<"}\n";
  }
  mapper.reset();
  if(rclcpp::ok())rclcpp::shutdown();
  return result;
}
