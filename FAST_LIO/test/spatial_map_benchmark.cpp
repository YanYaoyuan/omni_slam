#include "fast_lio/spatial_map.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

int main(int argc,char **argv) {
  if (argc!=7) {
    std::cerr<<"usage: spatial_map_benchmark PCD CACHE_DIR X Y Z RADIUS\n"; return 2;
  }
  try {
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();
    fast_lio::SpatialMap map(argv[1],argv[2],fast_lio::SpatialMap::Config{});
    const auto indexed=Clock::now();
    const Eigen::Vector3d center(std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5]));
    const double radius=std::stod(argv[6]);
    const auto region=map.region(center,radius); const auto loaded=Clock::now();
    const auto preview=map.preview(); const auto displayed=Clock::now();
    const auto repeat=map.region(center,radius); const auto finished=Clock::now();
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
    const auto stats=map.stats();
    auto ms=[](auto a,auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    std::cout<<"{\"file_bytes\":"<<file.tellg()<<",\"input_points\":"<<stats.input_points
      <<",\"filtered_points\":"<<stats.filtered_points<<",\"tiles\":"<<stats.tiles
      <<",\"reused_index\":"<<(stats.reused_index?"true":"false")
      <<",\"index_ms\":"<<ms(start,indexed)<<",\"region_ms\":"<<ms(indexed,loaded)
      <<",\"preview_ms\":"<<ms(loaded,displayed)<<",\"repeat_region_ms\":"<<ms(displayed,finished)
      <<",\"region_points\":"<<region.size()<<",\"preview_points\":"<<preview.size()
      <<",\"repeat_points\":"<<repeat.size()<<",\"cached_points\":"<<stats.cached_points
      <<",\"peak_rss_kib\":"<<usage.ru_maxrss<<"}\n";
    return 0;
  } catch (const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
}
