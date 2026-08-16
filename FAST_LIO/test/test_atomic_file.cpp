#include "fast_lio/atomic_file.hpp"

#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <unistd.h>

namespace
{

std::string make_test_directory()
{
  std::string pattern = "/tmp/fast_lio_atomic_file_XXXXXX";
  EXPECT_NE(::mkdtemp(&pattern[0]), nullptr);
  return pattern;
}

void write_text(const std::string &path, const std::string &value)
{
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(stream.good());
  stream << value;
  ASSERT_TRUE(stream.good());
}

std::string read_text(const std::string &path)
{
  std::ifstream stream(path, std::ios::binary);
  return std::string(
    std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

TEST(AtomicFile, ReplacesTargetOnlyAfterSuccessfulWrite)
{
  const std::string directory = make_test_directory();
  const std::string target = directory + "/map.pcd";
  write_text(target, "old-map");

  std::string error;
  ASSERT_TRUE(fast_lio::atomic_replace_file(
    target,
    [](const std::string &temporary) {
      write_text(temporary, "new-map");
      return true;
    },
    &error)) << error;
  EXPECT_EQ(read_text(target), "new-map");

  ::unlink(target.c_str());
  ::rmdir(directory.c_str());
}

TEST(AtomicFile, PreservesPreviousTargetWhenWriterFails)
{
  const std::string directory = make_test_directory();
  const std::string target = directory + "/map.pcd";
  write_text(target, "known-good-map");

  std::string error;
  EXPECT_FALSE(fast_lio::atomic_replace_file(
    target,
    [](const std::string &temporary) {
      write_text(temporary, "partial-map");
      return false;
    },
    &error));
  EXPECT_EQ(read_text(target), "known-good-map");

  ::unlink(target.c_str());
  ::rmdir(directory.c_str());
}

TEST(AtomicFile, RejectsWriterThatDoesNotCreateTemporaryFile)
{
  const std::string directory = make_test_directory();
  const std::string target = directory + "/map.pcd";

  std::string error;
  EXPECT_FALSE(fast_lio::atomic_replace_file(
    target, [](const std::string &) { return true; }, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(::access(target.c_str(), F_OK), -1);

  ::rmdir(directory.c_str());
}

}  // namespace
