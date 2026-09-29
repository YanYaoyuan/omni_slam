#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>

namespace fast_lio {
using DiskPoint=std::array<float,8>;
// Callback receives x/y/z/intensity/normal_x/normal_y/normal_z/curvature.
// A missing optional field is zero. compressed PCD uses a temporary disk stream,
// not a full-size decompression buffer. Temporary output belongs to the caller.
std::uint64_t streamPcd(const std::string &path, const std::string &scratch,
                       const std::function<void(const DiskPoint &)> &consume,
                       const std::function<bool()> &cancel);
}
