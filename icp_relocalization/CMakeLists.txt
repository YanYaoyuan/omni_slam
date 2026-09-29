cmake_minimum_required(VERSION 3.5)
project(icp_relocalization)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wextra -Wpedantic)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")
set(CMAKE_BUILD_TYPE Release)

option(USE_LIVOX "Build Livox CustomMsg support" OFF)
# define use_livox
if(USE_LIVOX)
  add_definitions(-DUSE_LIVOX)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(fast_lio REQUIRED)

# Ubuntu 22.04 exports Qhull::qhull_r, but the ROS Humble pcl_ros package
# refers to PCL's QHULL::QHULL compatibility target.  Imported CMake targets
# are scoped per package, so define the bridge here as well as in fast_lio.
find_package(Qhull CONFIG QUIET)
if(TARGET Qhull::qhull_r AND NOT TARGET QHULL::QHULL)
  add_library(QHULL::QHULL INTERFACE IMPORTED)
  set_property(TARGET QHULL::QHULL PROPERTY
    INTERFACE_LINK_LIBRARIES Qhull::qhull_r
  )
endif()

find_package(PCL REQUIRED COMPONENTS
  common
  io
  filters
  registration
  features
  search
  kdtree
  sample_consensus
)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
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
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3)
if(USE_LIVOX)
  find_package(livox_ros_driver2 REQUIRED)
endif()

include_directories(${OpenMP_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

add_executable(icp_node src/icp_node.cpp)
ament_target_dependencies(
  icp_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  fast_lio
  pcl_ros
  pcl_conversions
)
target_link_libraries(
  icp_node
  fast_lio::fastlio_spatial_map
  pcl_common
  pcl_io
  pcl_filters
  pcl_registration
  pcl_search
  pcl_kdtree
  pcl_sample_consensus
)
if(CMAKE_CROSSCOMPILING)
  target_link_options(icp_node PRIVATE
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()
if(USE_LIVOX)
  ament_target_dependencies(icp_node livox_ros_driver2)
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(
    test_relocalization_convergence
    test/test_relocalization_convergence.cpp
  )
  target_include_directories(
    test_relocalization_convergence PRIVATE include ${EIGEN3_INCLUDE_DIR})
endif()

add_executable(global_pointcloud_publisher src/global_pointcloud_publisher.cpp)
ament_target_dependencies(
  global_pointcloud_publisher
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(sac_ia_gicp
  src/sac_ia_gicp.cpp
)
ament_target_dependencies(
  sac_ia_gicp
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  pcl_ros
  pcl_conversions
  Eigen3
)
target_link_libraries(
  sac_ia_gicp
  pcl_common
  pcl_io
  pcl_filters
  pcl_registration
  pcl_features
  pcl_search
  pcl_kdtree
  pcl_sample_consensus
)
if(CMAKE_CROSSCOMPILING)
  target_link_options(sac_ia_gicp PRIVATE
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()

# Install launch files
install(
  DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

# Install nodes
install(
  TARGETS
    icp_node
    global_pointcloud_publisher
    sac_ia_gicp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)

ament_package()
