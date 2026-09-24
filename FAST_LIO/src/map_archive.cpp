#include "fast_lio/map_archive.hpp"
#include "fast_lio/atomic_file.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unistd.h>

namespace fast_lio {
namespace {
using PackedPoint = std::array<float, 8>;
static_assert(sizeof(PackedPoint) == 32, "PCD record must contain eight float32 fields");
PackedPoint pack(const pcl::PointXYZINormal &p) {
  return {{p.x, p.y, p.z, p.intensity, p.normal_x, p.normal_y, p.normal_z, p.curvature}};
}
pcl::PointXYZINormal unpack(const PackedPoint &p) {
  pcl::PointXYZINormal result{};
  result.x=p[0]; result.y=p[1]; result.z=p[2]; result.intensity=p[3];
  result.normal_x=p[4]; result.normal_y=p[5]; result.normal_z=p[6]; result.curvature=p[7];
  return result;
}
PackedPoint transform(PackedPoint p, const MapArchive::Pose &pose) {
  const Eigen::Vector3d xyz = pose * Eigen::Vector3d(p[0], p[1], p[2]);
  const Eigen::Vector3d normal = pose.linear() * Eigen::Vector3d(p[4], p[5], p[6]);
  for (int i=0; i<3; ++i) { p[i]=static_cast<float>(xyz[i]); p[i+4]=static_cast<float>(normal[i]); }
  return p;
}
MapArchive::Pose poseFor(const MapArchive::Frame &frame, const MapArchive::Poses &poses) {
  return frame.keyframe < poses.size()
      ? poses[frame.keyframe] * frame.keyframe_from_frame : frame.odom_pose;
}
}

struct MapArchive::Impl {
  struct Sample { PackedPoint point; std::size_t frame; };
  mutable std::mutex mutex;
  std::string root;
  std::size_t points_per_chunk, preview_limit;
  std::uint64_t count{0}, saved_count{0};
  bool saved{false}, discarded{false}, failed{false};
  Frames frames;
  std::vector<std::string> chunks;
  std::ofstream writer;
  std::vector<Sample> samples;
  std::mt19937_64 random{42};

  void write(const PackedPoint &point, std::size_t frame) {
    if (count % points_per_chunk == 0) {
      writer.close();
      const auto file = root + "/chunk-" + std::to_string(chunks.size()) + ".bin";
      writer.open(file, std::ios::binary | std::ios::trunc);
      if (!writer) throw std::runtime_error("cannot create map chunk: " + file);
      chunks.push_back(file);
    }
    writer.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
    if (!writer) throw std::runtime_error("writing dense map chunk failed (disk full or I/O error)");
    ++count;
    if (samples.size() < preview_limit) samples.push_back({point, frame});
    else if (preview_limit) {
      const auto replacement = std::uniform_int_distribution<std::uint64_t>(0, count-1)(random);
      if (replacement < preview_limit) samples[replacement] = {point, frame};
    }
  }

  template<class Consumer>
  void read(std::uint64_t first, std::uint64_t length, Consumer consume) const {
    // Only consume committed frame ranges; appending to the final chunk cannot
    // change those bytes. Chunk filenames are deterministic, no shared handle.
    std::array<PackedPoint, 4096> buffer;
    while (length) {
      const auto index = first / points_per_chunk;
      const auto offset = first % points_per_chunk;
      const auto take = std::min<std::uint64_t>(length, points_per_chunk-offset);
      std::ifstream input(root + "/chunk-" + std::to_string(index) + ".bin", std::ios::binary);
      if (!input) throw std::runtime_error("map chunk is missing or unreadable");
      input.seekg(static_cast<std::streamoff>(offset * sizeof(PackedPoint)));
      for (std::uint64_t done=0; done<take;) {
        const auto n = std::min<std::uint64_t>(buffer.size(), take-done);
        input.read(reinterpret_cast<char *>(buffer.data()), n*sizeof(PackedPoint));
        if (!input) throw std::runtime_error("map chunk is truncated");
        consume(buffer.data(), static_cast<std::size_t>(n));
        done += n;
      }
      first += take;
      length -= take;
    }
  }
};

MapArchive::MapArchive(const std::string &parent, std::size_t chunk_bytes, std::size_t preview_points)
  : impl_(new Impl) {
  if (chunk_bytes < sizeof(PackedPoint) || chunk_bytes > (64U << 20))
    throw std::invalid_argument("dense chunk size must be 32 bytes..64 MiB");
  impl_->points_per_chunk = chunk_bytes / sizeof(PackedPoint);
  impl_->preview_limit = preview_points;
  std::string pattern = parent + "/dense-session-XXXXXX";
  std::vector<char> writable(pattern.begin(), pattern.end()); writable.push_back('\0');
  const char *created = ::mkdtemp(writable.data());
  if (!created) throw std::runtime_error("cannot create dense map session in " + parent);
  impl_->root = created;
}
MapArchive::~MapArchive() = default;

std::size_t MapArchive::append(const Cloud &scan, const Pose &pose, double stamp,
                             std::size_t keyframe, const Pose &relative) {
  if (!pose.matrix().allFinite() || !relative.matrix().allFinite() || !std::isfinite(stamp))
    throw std::invalid_argument("non-finite dense scan pose/time");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->failed || impl_->discarded) throw std::runtime_error("dense archive is no longer writable");
  Frame frame;
  frame.first_point=impl_->count; frame.keyframe=keyframe; frame.stamp=stamp;
  frame.odom_pose=pose; frame.keyframe_from_frame=relative;
  const auto id = impl_->frames.size();
  try {
    for (const auto &p: scan) {
      if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) impl_->write(pack(p), id);
    }
    impl_->writer.flush();
    if (!impl_->writer && impl_->count != frame.first_point) throw std::runtime_error("flushing dense map failed");
  } catch (...) { impl_->failed = true; throw; }
  frame.point_count = impl_->count - frame.first_point;
  impl_->frames.push_back(frame);
  return id;
}
MapArchive::Frames MapArchive::frames() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->frames;
}
MapArchive::Cloud MapArchive::readFrame(std::size_t index) const {
  Frame frame;
  { std::lock_guard<std::mutex> lock(impl_->mutex); frame = impl_->frames.at(index); }
  Cloud cloud; cloud.reserve(frame.point_count);
  impl_->read(frame.first_point, frame.point_count, [&cloud](const PackedPoint *p, std::size_t count) {
    for (std::size_t i=0; i<count; ++i) cloud.push_back(unpack(p[i]));
  });
  return cloud;
}
MapArchive::Cloud MapArchive::preview(const Poses &poses, std::size_t frame_count) const {
  std::vector<Impl::Sample> samples;
  Frames snapshot;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->failed) throw std::runtime_error("dense archive write previously failed");
    samples = impl_->samples; snapshot=impl_->frames;
  }
  Cloud cloud; cloud.reserve(samples.size());
  for (const auto &sample: samples) {
    if (sample.frame >= frame_count) continue;
    cloud.push_back(unpack(transform(sample.point, poseFor(snapshot.at(sample.frame), poses))));
  }
  return cloud;
}
bool MapArchive::exportPcd(const std::string &path, const Poses &poses, std::string *error,
                          const std::function<bool()> &cancel, std::size_t frame_count,
                          double minimum_frame_interval) {
  if (!std::isfinite(minimum_frame_interval) || minimum_frame_interval<0) {
    if (error) *error="minimum frame interval must be finite and non-negative";
    return false;
  }
  Frames snapshot;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->failed || impl_->discarded || !impl_->count) {
      if (error) *error="dense archive is empty, discarded, or has a write failure";
      return false;
    }
    const auto n = std::min(frame_count, impl_->frames.size());
    if (!n) { if (error) *error="no committed frames to export"; return false; }
    snapshot.assign(impl_->frames.begin(), impl_->frames.begin()+n);
  }
  const std::uint64_t archived = snapshot.back().first_point + snapshot.back().point_count;
  std::vector<std::size_t> selected;
  selected.reserve(snapshot.size());
  double last_stamp=-std::numeric_limits<double>::infinity();
  for (std::size_t i=0; i<snapshot.size(); ++i) {
    const bool final=i+1==snapshot.size();
    if (selected.empty() || snapshot[i].stamp-last_stamp+1e-9>=minimum_frame_interval || final) {
      if (selected.empty() || selected.back()!=i) selected.push_back(i);
      last_stamp=snapshot[i].stamp;
    }
  }
  std::uint64_t total=0;
  for (const auto index:selected) total+=snapshot[index].point_count;
  bool written = false;
  try {
    written = atomic_replace_file(path, [&](const std::string &temporary) {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      output << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity normal_x normal_y normal_z curvature\n"
        "SIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\nWIDTH " << total <<
        "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << total << "\nDATA binary\n";
      try {
        for (const auto index: selected) {
          const auto &frame=snapshot[index];
          const auto pose = poseFor(frame, poses);
          impl_->read(frame.first_point, frame.point_count, [&](const PackedPoint *p, std::size_t n) {
            if (cancel && cancel()) throw std::runtime_error("map export canceled");
            std::array<PackedPoint, 4096> transformed;
            for (std::size_t i=0; i<n; ++i) {
              transformed[i] = transform(p[i], pose);
            }
            output.write(reinterpret_cast<const char *>(transformed.data()), n*sizeof(PackedPoint));
            if (!output) throw std::runtime_error("map export write failed");
          });
        }
        if (cancel && cancel()) throw std::runtime_error("map export canceled");
        output.flush();
        return static_cast<bool>(output);
      } catch (const std::exception &e) { if (error) *error=e.what(); return false; }
    }, error);
  } catch (const std::exception &e) { if (error) *error=e.what(); }
  if (written) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->saved=true; impl_->saved_count=archived;
  }
  return written;
}
std::uint64_t MapArchive::pointCount() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->count; }
std::string MapArchive::directory() const { return impl_->root; }
bool MapArchive::discardSavedChunks() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->saved || impl_->failed || impl_->saved_count != impl_->count) return false;
  impl_->writer.close();
  for (const auto &file: impl_->chunks) if (::unlink(file.c_str()) != 0) return false;
  if (::rmdir(impl_->root.c_str()) != 0) return false;
  impl_->discarded=true;
  return true;
}
}  // namespace fast_lio
