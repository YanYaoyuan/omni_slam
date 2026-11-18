#include <cstdint>
#include <cmath>
#include <queue>
#include <tuple>
#include <vector>
#include <limits>
#include <functional>

// Two nearest distinct seed groups on a fixed nonnegative CSR graph.
// Each accepted (vertex, group) state expands once. No label-specific edges.
extern "C" int nearest_two_groups(
    int64_t n, const int64_t* ptr, const int64_t* col, const double* weight,
    int64_t ns, const int64_t* sn, const int64_t* sg,
    int64_t* groups, double* distance) {
  try {
    const double inf = std::numeric_limits<double>::infinity();
    for (int64_t i=0;i<2*n;++i) { groups[i]=-1; distance[i]=inf; }
    using State = std::tuple<double,int64_t,int64_t>;
    std::priority_queue<State,std::vector<State>,std::greater<State>> q;
    for (int64_t i=0;i<ns;++i) q.emplace(0.,sn[i],sg[i]);
    while (!q.empty()) {
      auto [d,u,g] = q.top(); q.pop();
      if (groups[2*u]==g || groups[2*u+1]==g || groups[2*u+1]>=0) continue;
      int slot = groups[2*u]<0 ? 0 : 1;
      groups[2*u+slot]=g; distance[2*u+slot]=d;
      for (int64_t e=ptr[u];e<ptr[u+1];++e) {
        int64_t v=col[e];
        if (groups[2*v]==g || groups[2*v+1]==g || groups[2*v+1]>=0) continue;
        q.emplace(d+weight[e],v,g);
      }
    }
    return 0;
  } catch (...) { return 1; }
}
