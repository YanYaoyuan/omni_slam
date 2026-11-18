#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

inline double stamp_seconds(const builtin_interfaces::msg::Time &stamp)
{
  return static_cast<double>(stamp.sec) + stamp.nanosec * 1e-9;
}

inline builtin_interfaces::msg::Time stamp_from_seconds(double seconds)
{
  if (!std::isfinite(seconds) || seconds < 0.0)
    throw std::invalid_argument("sensor timestamp must be finite and nonnegative");
  return rclcpp::Time(static_cast<int64_t>(std::llround(seconds * 1e9)));
}

inline geometry_msgs::msg::Quaternion quaternion_from_rpy(double roll, double pitch, double yaw)
{
  const Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
  geometry_msgs::msg::Quaternion out;
  out.x = q.x(); out.y = q.y(); out.z = q.z(); out.w = q.w();
  return out;
}

template <typename T, typename Default>
void read_parameter(rclcpp::Node &node, std::string name, T &value, const Default &fallback)
{
  std::replace(name.begin(), name.end(), '/', '.');
  if constexpr (std::is_same_v<T, std::vector<int>>) {
    const std::vector<int64_t> defaults(fallback.begin(), fallback.end());
    if (!node.has_parameter(name)) node.declare_parameter<std::vector<int64_t>>(name, defaults);
    const auto values = node.get_parameter(name).as_integer_array();
    value.assign(values.begin(), values.end());
  } else {
    if (!node.has_parameter(name)) node.declare_parameter<T>(name, static_cast<T>(fallback));
    value = node.get_parameter(name).get_value<T>();
  }
}

inline std::string &runtime_directory()
{
  static std::string directory = "/tmp/fast_livo/";
  return directory;
}
