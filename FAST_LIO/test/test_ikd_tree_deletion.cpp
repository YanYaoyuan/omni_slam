#include "ikd-Tree/ikd_Tree.h"
#include <gtest/gtest.h>
#include <set>
#include <tuple>

TEST(IkdTreeDeletion, RemovesPointsOnTiedSplitPlanesWithoutRemovingRetainedRegion)
{
  using Tree = KD_TREE<pcl::PointXYZINormal>;
  // Avoid asynchronous rebuilding here to isolate the split-plane lookup.
  auto owner = std::make_unique<Tree>(.99f, .99f, .3f);
  auto &tree = *owner;
  Tree::PointVector all, departing, retained;
  for (int x = 0; x < 12; ++x) for (int y = 0; y < 12; ++y)
    for (int z = 0; z < 5; ++z) {
      pcl::PointXYZINormal p{};
      p.x = x; p.y = y; p.z = z;
      all.push_back(p);
      if (x < 6) departing.push_back(p); else retained.push_back(p);
    }
  tree.Build(all);
  tree.Delete_Points(departing);
  Tree::PointVector actual;
  tree.flatten(tree.Root_Node, actual, NOT_RECORD);
  const auto records = [](const Tree::PointVector & points) {
    std::set<std::tuple<float, float, float>> result;
    for (const auto &p : points) result.emplace(p.x, p.y, p.z);
    return result;
  };
  EXPECT_EQ(records(actual), records(retained));
  EXPECT_EQ(actual.size(), retained.size());
  tree.Delete_Points(retained);
  actual.clear(); tree.flatten(tree.Root_Node, actual, NOT_RECORD);
  EXPECT_TRUE(actual.empty());
}
