cmake_minimum_required(VERSION 3.8)
project(fast_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

ADD_COMPILE_OPTIONS(-std=c++14)
ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch64)|(AARCH64)|(arm64)|(ARM64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")

  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Ubuntu 22.04's Qhull config exports Qhull::qhull_r, while the ROS Humble
# pcl_ros Debian export refers to PCL's compatibility target QHULL::QHULL.
# Depending on CMake's config/module lookup order, PCL may not create that
# compatibility target itself, which then breaks generation of every target
# using pcl_ros.  Define the missing bridge before importing pcl_ros.
find_package(Qhull CONFIG QUIET)
if(TARGET Qhull::qhull_r AND NOT TARGET QHULL::QHULL)
  add_library(QHULL::QHULL INTERFACE IMPORTED)
  set_property(TARGET QHULL::QHULL PROPERTY
    INTERFACE_LINK_LIBRARIES Qhull::qhull_r
  )
endif()

find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# The upstream ARM64 pcl_ros Debian export captures every PCL component and
# native /usr include path.  In a sysroot build that incorrectly pulls in the
# unused VTK/X11 visualization stack.  Retain the ROS/tf2 and PCL components
# used by pcl_ros_tf while keeping native builds untouched.
if(CMAKE_CROSSCOMPILING AND TARGET pcl_ros::pcl_ros_tf)
  set_target_properties(pcl_ros::pcl_ros_tf PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
      "/opt/ros/humble/include/pcl_ros;/opt/ros/humble/include/pcl_conversions;${PCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES
      "rclcpp::rclcpp;tf2::tf2;tf2_ros::tf2_ros;tf2_geometry_msgs::tf2_geometry_msgs;pcl_common"
  )
endif()
if(CMAKE_CROSSCOMPILING AND TARGET pcl_ros::pcd_to_pointcloud_lib)
  set_target_properties(pcl_ros::pcd_to_pointcloud_lib PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
      "/opt/ros/humble/include/pcl_ros;/opt/ros/humble/include/pcl_conversions;${PCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "rclcpp::rclcpp;pcl_common;pcl_io"
  )
endif()

option(USE_LIVOX_CUSTOM_MSG "Build Livox CustomMsg support" OFF)
if(USE_LIVOX_CUSTOM_MSG)
  find_package(livox_ros_driver2 REQUIRED)
  add_definitions(-DUSE_LIVOX)
endif()

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

if(USE_LIVOX_CUSTOM_MSG)
  list(APPEND dependencies livox_ros_driver2)
endif()

# Thirdparty libraries
find_package(Eigen3 REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
message(STATUS "PCL: ${PCL_INCLUDE_DIRS}")

set(msg_files
  "msg/Pose6D.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)

if(CMAKE_CROSSCOMPILING)
  target_link_options(fastlio_mapping PRIVATE
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  fastlio_mapping
  pcl_common
  pcl_io
  pcl_filters
  pcl_kdtree
  ${PYTHON_LIBRARIES}
  Eigen3::Eigen
)
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")

if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  # Custommsg to support foxy & galactic
  rosidl_target_interfaces(fastlio_mapping
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
endif()

ament_target_dependencies(fastlio_mapping ${dependencies})

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_atomic_file test/test_atomic_file.cpp)
  target_include_directories(test_atomic_file PRIVATE include)
  ament_add_gtest(test_extent_safe_voxel_filter test/test_extent_safe_voxel_filter.cpp)
  target_include_directories(test_extent_safe_voxel_filter PRIVATE include ${PCL_INCLUDE_DIRS})
  target_link_libraries(test_extent_safe_voxel_filter pcl_common)
  ament_add_gtest(test_reference_odometry test/test_reference_odometry.cpp)
  target_include_directories(test_reference_odometry PRIVATE include)
  target_link_libraries(test_reference_odometry Eigen3::Eigen)
  ament_add_pytest_test(phase0_regressions
    test/test_phase0_regressions.py
  )
  ament_add_pytest_test(phase2_regressions
    test/test_phase2_regressions.py
  )
endif()

# ROSIDL 会把本包的公共 include 根导出为 include/fast_lio。公共工具头也安装
# 到同一包命名空间下，保证下游可以稳定使用 <fast_lio/...>，且不会污染
# 工作区的全局 include 根。
install(DIRECTORY include/fast_lio/ DESTINATION include/${PROJECT_NAME}/fast_lio)
ament_export_include_directories(include/${PROJECT_NAME})

ament_package()
