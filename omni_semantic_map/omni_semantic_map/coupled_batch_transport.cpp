#include <cstdint>
#include <cmath>
#include <queue>
#include <tuple>
#include <vector>
#include <limits>
#include <functional>

// A frame stores at most two different owners at its shortest distance (within
// tolerance). Two conflicting owners suffice to establish ambiguity. Retain
// all frames within tolerance of the second nearest distinct frame, including
// ties. Pruning at a vertex is safe: two strictly closer distinct frames also
// dominate any continuation after removal of any one frame.
struct FrameState {
  int64_t batch;
  int64_t owner[2];
  double distance[2];
  int count;
};

static int64_t read_owner(const std::vector<FrameState>& frames,
                          int64_t excluded, double tolerance, double& minimum) {
  const double inf = std::numeric_limits<double>::infinity();
  minimum = inf;
  int64_t best = -1;
  for (const auto& frame : frames) {
    if (frame.batch == excluded) continue;
    for (int j = 0; j < frame.count; ++j) {
      if (frame.distance[j] < minimum ||
          (frame.distance[j] == minimum && frame.owner[j] < best)) {
        minimum = frame.distance[j]; best = frame.owner[j];
      }
    }
  }
  if (best < 0) return -1;
  double other = inf;
  for (const auto& frame : frames) {
    if (frame.batch == excluded) continue;
    for (int j = 0; j < frame.count; ++j)
      if (frame.owner[j] != best) other = std::min(other, frame.distance[j]);
  }
  return minimum < other - tolerance ? best : -1;
}

extern "C" int coupled_batch_transport(
    int64_t n, const int64_t* ptr, const int64_t* col, const double* weight,
    int64_t ns, const int64_t* sn, const int64_t* so, const int64_t* sb,
    double tolerance, int64_t* before, int64_t* after, int64_t* critical,
    double* nearest_distance, double* after_distance, int64_t* stats) {
  try {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<FrameState>> records(n);
    std::vector<double> second(n, inf);
    using State = std::tuple<double, int64_t, int64_t, int64_t>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> q;
    stats[0] = stats[1] = stats[2] = 0;
    for (int64_t i=0; i<ns; ++i) q.emplace(0., sn[i], sb[i], so[i]);
    while (!q.empty()) {
      auto [distance, node, batch, owner] = q.top(); q.pop(); ++stats[0];
      if (distance > second[node] + tolerance) continue;
      auto& frames = records[node];
      FrameState* entry = nullptr;
      for (auto& frame : frames) if (frame.batch == batch) { entry = &frame; break; }
      if (entry) {
        if (entry->count == 2 || entry->owner[0] == owner ||
            distance > entry->distance[0] + tolerance) continue;
        entry->owner[1] = owner; entry->distance[1] = distance; entry->count = 2;
      } else {
        frames.push_back({batch, {owner, -1}, {distance, inf}, 1});
        if (frames.size() == 2) second[node] = distance;
      }
      ++stats[1];
      stats[2] = std::max(stats[2], static_cast<int64_t>(frames.size()));
      for (int64_t e=ptr[node]; e<ptr[node+1]; ++e) {
        int64_t next = col[e]; double nd = distance + weight[e];
        if (nd <= second[next] + tolerance) q.emplace(nd, next, batch, owner);
      }
    }
    for (int64_t node=0; node<n; ++node) {
      critical[node] = records[node].empty() ? -1 : records[node][0].batch;
      before[node] = read_owner(records[node], -1, tolerance, nearest_distance[node]);
      after[node] = read_owner(records[node], critical[node], tolerance, after_distance[node]);
    }
    return 0;
  } catch (...) { return 1; }
}
