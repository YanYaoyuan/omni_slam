cmake_minimum_required(VERSION 3.8)
project(omni_slam_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SlamStatus.msg"
  "srv/StartMapping.srv"
  "srv/StopMapping.srv"
  "srv/StopLocalization.srv"
  "action/SaveMap.action"
  "action/StartLocalization.action"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
