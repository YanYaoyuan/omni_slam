// Accelerated storage/graph benchmark; live 5 Hz timing is measured separately
// by replay_slam.py. Repeated real scans are deliberately synthetic test data.
#include "fast_lio/loop_closure_backend.hpp"
#include "../src/pcd_stream.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
double percentile(std::vector<double> values,double fraction) {
  if (values.empty()) return 0;
  std::sort(values.begin(),values.end());
  return values[std::min(values.size()-1,static_cast<std::size_t>(fraction*(values.size()-1)))];
}
}
int main(int argc,char **argv) {
  if (argc!=4) { std::cerr<<"usage: map_pipeline_benchmark SOURCE_PCD OUTPUT_DIR MIB\n";return 2; }
  try {
    namespace fs=std::filesystem;
    using Backend=fast_lio::LoopClosureBackend;
    const std::uint64_t bytes=std::stoull(argv[3])*(1ULL<<20),points=bytes/32;
    const fs::path directory=argv[2];
    if (fs::exists(directory)) throw std::runtime_error("output directory already exists");
    fs::create_directories(directory);
    Backend::Cloud scan;
    fast_lio::streamPcd(argv[1],(directory/"reader.bin").string(),[&](const fast_lio::DiskPoint &p) {
      if (scan.size()>=8880 || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) return;
      pcl::PointXYZINormal point{};point.x=p[0];point.y=p[1];point.z=p[2];point.intensity=p[3];
      scan.push_back(point);
    },{});
    if (scan.size()<1000) throw std::runtime_error("source scan too small");
    Backend backend(directory.string(),Backend::Config{},false);
    std::vector<double> appends,optimizations,previews;
    std::uint64_t written=0;std::size_t frames=0,max_backlog=0,preview_points=0;
    const auto begin=Clock::now();
    auto drain=[&] {
      const auto before=backend.snapshot();
      max_backlog=std::max(max_backlog,before.keyframes.size()-before.processed_keyframes);
      auto start=Clock::now();backend.processPending();optimizations.push_back(elapsed(start));
      const auto after=backend.snapshot();
      if (!after.error.empty()) throw std::runtime_error(after.error);
      if (after.processed_keyframes!=after.keyframes.size()) throw std::runtime_error("undrained graph");
      start=Clock::now(); const auto cloud=backend.preview(after);previews.push_back(elapsed(start));
      preview_points=std::max(preview_points,cloud.size());
      if (cloud.size()>300000) throw std::runtime_error("unbounded preview");
    };
    while (written<points) {
      const auto take=std::min<std::uint64_t>(scan.size(),points-written);
      Backend::Cloud partial;
      if (take<scan.size()) partial.assign(scan.begin(),scan.begin()+take);
      auto odom=Backend::Pose::Identity();odom.translation().x()=.8*double(written)/points;
      const auto start=Clock::now();backend.append(take==scan.size()?scan:partial,odom,frames*.2);
      appends.push_back(elapsed(start));written+=take;++frames;
      // Accelerated input is drained explicitly every five simulated seconds,
      // allowing graph cost and input-call latency to be reported separately.
      if (frames%25==0) drain();
    }
    // Reobserve the same scene after enough time to validate a real ICP closure
    // even for the 30 MiB sample, whose preceding scans span less than 30 s.
    auto returned=Backend::Pose::Identity();returned.translation().x()=.8;
    const auto last=Clock::now();backend.append(scan,returned,frames*.2+35);
    appends.push_back(elapsed(last));written+=scan.size();++frames;drain();
    const auto state=backend.snapshot();
    if (!state.loop_count) throw std::runtime_error("no validated ICP loop in benchmark");
    if (std::abs(state.frames.back().odom_pose.translation().x()-.8)>1e-9)
      throw std::runtime_error("loop changed stored local odometry");
    std::uintmax_t max_chunk=0;
    for (const auto &file:fs::directory_iterator(backend.directory()))
      if (file.path().extension()==".bin") max_chunk=std::max(max_chunk,file.file_size());
    if (max_chunk>(64U<<20)) throw std::runtime_error("unbounded archive chunk");
    std::string error;const auto save_start=Clock::now();
    if (!backend.save((directory/"map.pcd").string(),&error)) throw std::runtime_error(error);
    const double save_ms=elapsed(save_start),total_ms=elapsed(begin);
    const auto actual=fs::file_size(directory/"map.pcd");
    if (actual<written*32 || actual>written*32+4096) throw std::runtime_error("PCD size mismatch");
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    if (!backend.discardSavedChunks()) throw std::runtime_error("saved chunks not released");
    std::cout<<"{\"synthetic\":true,\"scope\":\"accelerated archive/preview/graph/save; not live LIO timing\","
      <<"\"requested_mib\":"<<argv[3]<<",\"file_bytes\":"<<actual<<",\"points\":"<<written
      <<",\"frames\":"<<frames<<",\"keys\":"<<state.keyframes.size()<<",\"loops\":"<<state.loop_count
      <<",\"max_pending_keys\":"<<max_backlog<<",\"preview_points\":"<<preview_points
      <<",\"max_chunk_bytes\":"<<max_chunk<<",\"append_p99_ms\":"<<percentile(appends,.99)
      <<",\"append_max_ms\":"<<percentile(appends,1)<<",\"graph_p99_ms\":"<<percentile(optimizations,.99)
      <<",\"graph_max_ms\":"<<percentile(optimizations,1)<<",\"preview_p99_ms\":"<<percentile(previews,.99)
      <<",\"save_ms\":"<<save_ms<<",\"total_ms\":"<<total_ms<<",\"peak_rss_kib\":"<<usage.ru_maxrss
      <<",\"cpu_seconds\":"<<usage.ru_utime.tv_sec+usage.ru_utime.tv_usec/1e6+usage.ru_stime.tv_sec+usage.ru_stime.tv_usec/1e6
      <<",\"final_correction_m\":"<<(state.keyframes.back().translation()-state.frames.back().odom_pose.translation()).norm()<<"}\n";
    return 0;
  } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
