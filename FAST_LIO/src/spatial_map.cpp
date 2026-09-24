#include "fast_lio/spatial_map.hpp"
#include "fast_lio/extent_safe_voxel_filter.hpp"
#include "pcd_stream.hpp"
#include <openssl/sha.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace fast_lio {
namespace {
namespace fs=std::filesystem;
using Tile=SpatialMap::Tile;
using Cloud=SpatialMap::Cloud;
using Key=pointcloud::detail::VoxelKey;
using KeyHash=pointcloud::detail::VoxelKeyHash;
std::string name(const Tile &t) {
  return std::to_string(t[0])+"_"+std::to_string(t[1])+"_"+std::to_string(t[2])+".bin";
}
std::string identity(const std::string &file,const SpatialMap::Config &config) {
  struct stat st{};
  if (::stat(file.c_str(),&st)!=0 || !S_ISREG(st.st_mode)) throw std::runtime_error("PCD is not a readable regular file");
  std::ostringstream text;
  text<<fs::canonical(file).string()<<'\n'<<st.st_size<<' '<<st.st_ino<<' '
      <<st.st_mtim.tv_sec<<' '<<st.st_mtim.tv_nsec<<' '
      <<std::setprecision(17)<<config.voxel_size<<' '<<config.tile_size<<' '<<config.preview_points;
  const auto s=text.str(); unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(s.data()),s.size(),hash);
  std::ostringstream result;
  for (auto c:hash) result<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
  return result.str();
}
struct FileLock {
  int fd{-1};
  FileLock(const std::string &path,const std::function<bool()> &cancel) {
    fd=::open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
    if (fd<0) throw std::runtime_error("cannot create map index lock");
    while (::flock(fd,LOCK_EX|LOCK_NB)!=0) {
      if ((errno!=EWOULDBLOCK && errno!=EAGAIN) || (cancel && cancel())) {
        ::close(fd); fd=-1; throw std::runtime_error("map index lock failed or canceled");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  ~FileLock() { if(fd>=0) { ::flock(fd,LOCK_UN); ::close(fd); } }
};
struct Scratch {
  fs::path path;
  explicit Scratch(const fs::path &root) {
    const auto pattern=(root/"building-XXXXXX").string();
    std::vector<char> chars(pattern.begin(),pattern.end()); chars.push_back(0);
    const char *created=::mkdtemp(chars.data());
    if (!created) throw std::runtime_error("cannot create map index scratch directory");
    path=created;
  }
  ~Scratch() { if(!path.empty()) { std::error_code error; fs::remove_all(path,error); } }
};
pcl::PointXYZINormal unpack(const DiskPoint &p) {
  pcl::PointXYZINormal result{};
  result.x=p[0]; result.y=p[1]; result.z=p[2]; result.intensity=p[3];
  result.normal_x=p[4]; result.normal_y=p[5]; result.normal_z=p[6]; result.curvature=p[7];
  return result;
}
void writePoint(std::ostream &out,const DiskPoint &p) {
  out.write(reinterpret_cast<const char *>(p.data()),sizeof(p));
  if (!out) throw std::runtime_error("map tile write failed");
}
template<class Visitor> void readPoints(const fs::path &file,Visitor visit) {
  std::ifstream in(file,std::ios::binary);
  if (!in) throw std::runtime_error("map tile is unreadable: "+file.string());
  std::array<DiskPoint,4096> buffer;
  while (in) {
    in.read(reinterpret_cast<char *>(buffer.data()),sizeof(buffer));
    const auto bytes=in.gcount();
    if (bytes%sizeof(DiskPoint)) throw std::runtime_error("truncated map tile");
    for (std::size_t i=0;i<static_cast<std::size_t>(bytes)/sizeof(DiskPoint);++i) visit(buffer[i]);
  }
  if (!in.eof()) throw std::runtime_error("map tile read failed");
}
std::int64_t floorDivide(std::int64_t value,std::int64_t divisor) {
  auto q=value/divisor;
  if (value%divisor<0) --q;
  return q;
}
}

struct SpatialMap::Impl {
  struct Cached { Cloud::ConstPtr cloud; std::list<Tile>::iterator recency; };
  Config config;
  Stats info;
  fs::path root;
  std::int64_t tile_voxels;
  std::map<Tile,std::uint64_t> counts;
  mutable std::mutex mutex;
  std::list<Tile> recency;
  std::map<Tile,Cached> cache;

  Impl(Config c):config(c) {
    if (!std::isfinite(c.voxel_size) || !std::isfinite(c.tile_size) || c.voxel_size<=0 ||
        c.tile_size<c.voxel_size || c.tile_size/c.voxel_size>128 ||
        !c.region_points || !c.preview_points || c.preview_points>300000)
      throw std::invalid_argument("invalid spatial map size/cache configuration");
    tile_voxels=static_cast<std::int64_t>(std::ceil(c.tile_size/c.voxel_size));
    // Align tile boundaries with the existing global voxel grid. For 0.3 m
    // voxels, nominal 20 m tiles span 20.1 m; no duplicate boundary voxels.
    info.tile_size=tile_voxels*c.voxel_size;
  }
  bool voxel(const DiskPoint &p,Key *key) const {
    return pointcloud::detail::coordinateToVoxelIndex(p[0],1/config.voxel_size,&key->x) &&
      pointcloud::detail::coordinateToVoxelIndex(p[1],1/config.voxel_size,&key->y) &&
      pointcloud::detail::coordinateToVoxelIndex(p[2],1/config.voxel_size,&key->z);
  }
  Tile tileKey(const Key &k) const {
    return {{floorDivide(k.x,tile_voxels),floorDivide(k.y,tile_voxels),floorDivide(k.z,tile_voxels)}};
  }
  void build(const std::string &pcd,const fs::path &stage,const std::function<bool()> &cancel) {
    fs::create_directory(stage/"raw"); fs::create_directory(stage/"tiles");
    struct Open { std::ofstream stream; std::uint64_t used; };
    std::map<Tile,Open> writers; std::uint64_t tick=0;
    info.input_points=streamPcd(pcd,(stage/"decompressed.bin").string(),[&](const DiskPoint &p) {
      Key key; if (!voxel(p,&key)) return;
      const auto tile=tileKey(key); ++counts[tile]; ++info.valid_points;
      if (counts.size()>250000) throw std::runtime_error("map has too many spatial tiles (check outliers)");
      auto entry=writers.find(tile);
      if (entry==writers.end()) {
        if (writers.size()>=16) {
          const auto oldest=std::min_element(writers.begin(),writers.end(),
            [](const auto &a,const auto &b) { return a.second.used<b.second.used; });
          oldest->second.stream.close();
          if (!oldest->second.stream) throw std::runtime_error("raw map tile flush failed");
          writers.erase(oldest);
        }
        entry=writers.emplace(tile,Open{}).first;
        entry->second.stream.open(stage/"raw"/name(tile),std::ios::binary|std::ios::app);
      }
      entry->second.used=++tick; writePoint(entry->second.stream,p);
    },cancel);
    for (auto &entry:writers) {
      entry.second.stream.close();
      if (!entry.second.stream) throw std::runtime_error("raw map tile flush failed");
    }
    writers.clear();
    if (!info.valid_points) throw std::runtime_error("map contains no usable finite points");
    std::vector<DiskPoint> preview; preview.reserve(config.preview_points);
    std::mt19937_64 random(42);
    for (auto &entry:counts) {
      if (cancel && cancel()) throw std::runtime_error("map indexing canceled");
      struct Best { DiskPoint point; double distance; };
      std::unordered_map<Key,Best,KeyHash> voxels;
      // Only one spatial tile's distinct voxels exist in RAM during indexing.
      readPoints(stage/"raw"/name(entry.first),[&](const DiskPoint &p) {
        Key key; if (!voxel(p,&key)) return;
        double distance=0;
        const std::array<std::int64_t,3> indexes{{key.x,key.y,key.z}};
        for (int axis=0;axis<3;++axis) {
          const double delta=p[axis]-(indexes[axis]+.5)*config.voxel_size;
          distance+=delta*delta;
        }
        const auto result=voxels.emplace(key,Best{p,distance});
        if (!result.second && distance<result.first->second.distance) result.first->second={p,distance};
      });
      std::vector<Key> ordered; ordered.reserve(voxels.size());
      for (const auto &v:voxels) ordered.push_back(v.first);
      std::sort(ordered.begin(),ordered.end(),[](const Key &a,const Key &b) {
        return std::tie(a.x,a.y,a.z)<std::tie(b.x,b.y,b.z);
      });
      std::ofstream out(stage/"tiles"/name(entry.first),std::ios::binary|std::ios::trunc);
      for (const auto &key:ordered) {
        const auto &point=voxels.at(key).point; writePoint(out,point); ++info.filtered_points;
        if (preview.size()<config.preview_points) preview.push_back(point);
        else {
          const auto replacement=std::uniform_int_distribution<std::uint64_t>(0,info.filtered_points-1)(random);
          if (replacement<preview.size()) preview[replacement]=point;
        }
      }
      out.close(); if (!out) throw std::runtime_error("filtered map tile flush failed");
      entry.second=ordered.size();
      fs::remove(stage/"raw"/name(entry.first));
    }
    fs::remove(stage/"raw"); fs::remove(stage/"decompressed.bin");
    std::ofstream preview_file(stage/"preview.bin",std::ios::binary|std::ios::trunc);
    for (const auto &p:preview) writePoint(preview_file,p);
    preview_file.close(); if(!preview_file) throw std::runtime_error("map preview flush failed");
    std::ofstream manifest(stage/"index.txt");
    manifest<<"HENG_PCD_TILES_1\n"<<std::setprecision(17)<<config.voxel_size<<' '<<info.tile_size<<' '
      <<info.input_points<<' '<<info.valid_points<<' '<<info.filtered_points<<' '<<counts.size()<<'\n';
    for (const auto &entry:counts)
      manifest<<entry.first[0]<<' '<<entry.first[1]<<' '<<entry.first[2]<<' '<<entry.second<<'\n';
    manifest.close(); if(!manifest) throw std::runtime_error("map index commit failed");
    info.tiles=counts.size();
  }
  void readIndex() {
    std::ifstream in(root/"index.txt"); std::string magic; double leaf=0,edge=0;
    std::size_t n=0;
    if (!(in>>magic>>leaf>>edge>>info.input_points>>info.valid_points>>info.filtered_points>>n) ||
        magic!="HENG_PCD_TILES_1" || std::abs(leaf-config.voxel_size)>1e-12 ||
        std::abs(edge-info.tile_size)>1e-9 || n>250000 || !n || !info.filtered_points)
      throw std::runtime_error("invalid spatial map cache index");
    std::uint64_t total=0;
    for (std::size_t i=0;i<n;++i) {
      Tile tile; std::uint64_t size;
      if (!(in>>tile[0]>>tile[1]>>tile[2]>>size) || !size ||
          size>static_cast<std::uint64_t>(tile_voxels*tile_voxels*tile_voxels) ||
          !counts.emplace(tile,size).second) throw std::runtime_error("invalid spatial tile entry");
      if (fs::file_size(root/"tiles"/name(tile))!=size*sizeof(DiskPoint))
        throw std::runtime_error("cached map tile is truncated");
      total+=size;
    }
    if (total!=info.filtered_points) throw std::runtime_error("map cache point counts disagree");
    info.tiles=n; info.reused_index=true;
  }
};

SpatialMap::SpatialMap(const std::string &pcd,const std::string &cache_root,Config config,
                       const std::function<bool()> &cancel):impl_(new Impl(config)) {
  fs::create_directories(cache_root);
  const auto fingerprint=identity(pcd,config); impl_->root=fs::path(cache_root)/fingerprint;
  FileLock lock(impl_->root.string()+".lock",cancel);
  if (fs::exists(impl_->root/"index.txt")) { impl_->readIndex(); return; }
  Scratch scratch(cache_root);
  impl_->build(pcd,scratch.path,cancel);
  if (identity(pcd,config)!=fingerprint) throw std::runtime_error("PCD changed while building spatial index");
  if (cancel && cancel()) throw std::runtime_error("map indexing canceled");
  fs::rename(scratch.path,impl_->root); scratch.path.clear();
}
SpatialMap::~SpatialMap()=default;
std::uint64_t SpatialMap::pointCount(const Tile &key) const { return impl_->counts.at(key); }
std::vector<SpatialMap::Tile> SpatialMap::tilesFor(const Eigen::Vector3d &center,double radius) const {
  if (!center.allFinite() || !std::isfinite(radius) || radius<=0) throw std::invalid_argument("invalid map region");
  std::vector<Tile> result;
  for (const auto &entry:impl_->counts) {
    bool intersects=true;
    for (int axis=0;axis<3;++axis) {
      const double low=entry.first[axis]*impl_->info.tile_size;
      intersects=intersects && low<=center[axis]+radius && low+impl_->info.tile_size>=center[axis]-radius;
    }
    if (intersects) result.push_back(entry.first);
  }
  return result;
}
SpatialMap::Cloud::ConstPtr SpatialMap::tile(const Tile &key) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto cached=impl_->cache.find(key);
  if (cached!=impl_->cache.end()) {
    impl_->recency.splice(impl_->recency.begin(),impl_->recency,cached->second.recency);
    return cached->second.cloud;
  }
  const auto count=impl_->counts.at(key);
  Cloud::Ptr result(new Cloud); result->reserve(count);
  readPoints(impl_->root/"tiles"/name(key),[&](const DiskPoint &p) {
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
      throw std::runtime_error("non-finite point in spatial map cache");
    result->push_back(unpack(p));
  });
  if (result->size()!=count) throw std::runtime_error("map tile size changed");
  while (impl_->info.cached_points+count>impl_->config.cache_points && !impl_->recency.empty()) {
    const auto oldest=impl_->cache.find(impl_->recency.back());
    impl_->info.cached_points-=oldest->second.cloud->size();
    impl_->cache.erase(oldest); impl_->recency.pop_back();
  }
  if (count<=impl_->config.cache_points) {
    impl_->recency.push_front(key); impl_->cache.emplace(key,Impl::Cached{result,impl_->recency.begin()});
    impl_->info.cached_points+=count;
  }
  return result;
}
SpatialMap::Cloud SpatialMap::region(const Eigen::Vector3d &center,double radius) {
  Cloud result;
  for (const auto &key:tilesFor(center,radius)) {
    const auto cloud=tile(key);
    for (const auto &point:*cloud) {
      if (std::abs(point.x-center.x())>radius || std::abs(point.y-center.y())>radius ||
          std::abs(point.z-center.z())>radius) continue;
      if (result.size()>=impl_->config.region_points) throw std::runtime_error("active map region exceeds configured point budget");
      result.push_back(point);
    }
  }
  return result;
}
SpatialMap::Cloud SpatialMap::preview() const {
  Cloud result;
  readPoints(impl_->root/"preview.bin",[&](const DiskPoint &p) {
    if (result.size()>=impl_->config.preview_points) throw std::runtime_error("cached map preview exceeds point budget");
    result.push_back(unpack(p));
  });
  return result;
}
SpatialMap::Stats SpatialMap::stats() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->info; }
std::string SpatialMap::directory() const { return impl_->root.string(); }
}  // namespace fast_lio
