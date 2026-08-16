#pragma once

#include <cerrno>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fast_lio
{

inline std::string parent_directory(const std::string &path)
{
  const std::size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return slash == 0 ? "/" : path.substr(0, slash);
}

inline bool sync_file(const std::string &path, std::string *error)
{
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (error) {
      *error = "open temporary file: " + std::string(std::strerror(errno));
    }
    return false;
  }
  const int result = ::fsync(fd);
  const int saved_errno = errno;
  ::close(fd);
  if (result != 0) {
    if (error) {
      *error = "fsync temporary file: " + std::string(std::strerror(saved_errno));
    }
    return false;
  }
  return true;
}

inline bool sync_directory(const std::string &path, std::string *error)
{
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    if (error) {
      *error = "open map directory: " + std::string(std::strerror(errno));
    }
    return false;
  }
  const int result = ::fsync(fd);
  const int saved_errno = errno;
  ::close(fd);
  if (result != 0) {
    if (error) {
      *error = "fsync map directory: " + std::string(std::strerror(saved_errno));
    }
    return false;
  }
  return true;
}

// The writer must create and completely write the supplied temporary path.
// The previous target remains untouched unless writing and fsync both succeed.
inline bool atomic_replace_file(
  const std::string &target,
  const std::function<bool(const std::string &)> &writer,
  std::string *error)
{
  if (target.empty()) {
    if (error) {
      *error = "target path is empty";
    }
    return false;
  }

  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string temporary =
    target + ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." +
    std::to_string(static_cast<long long>(nonce));

  ::unlink(temporary.c_str());
  if (!writer(temporary)) {
    ::unlink(temporary.c_str());
    if (error && error->empty()) {
      *error = "temporary file writer failed";
    }
    return false;
  }
  if (!sync_file(temporary, error)) {
    ::unlink(temporary.c_str());
    return false;
  }
  if (::rename(temporary.c_str(), target.c_str()) != 0) {
    const int saved_errno = errno;
    ::unlink(temporary.c_str());
    if (error) {
      *error = "rename temporary map: " + std::string(std::strerror(saved_errno));
    }
    return false;
  }
  return sync_directory(parent_directory(target), error);
}

}  // namespace fast_lio
