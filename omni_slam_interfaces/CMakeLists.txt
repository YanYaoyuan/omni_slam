cmake_minimum_required(VERSION 3.8)
project(omni_slam_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LivoKeyframe.msg"
  "msg/LivoSemanticFrame.msg"
  "srv/StartMapping.srv"
  "srv/StopMapping.srv"
  "srv/StopLocalization.srv"
  "action/SaveMap.action"
  "action/StartLocalization.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
