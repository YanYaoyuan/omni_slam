from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_root = LaunchConfiguration("map_root")
    map_source_path = LaunchConfiguration("map_source_path")
    return LaunchDescription(
        [
            DeclareLaunchArgument("map_root", default_value="/var/lib/omni/maps"),
            DeclareLaunchArgument(
                "map_source_path", default_value="/tmp/omni_slam_work/map.pcd"
            ),
            Node(
                package="omni_slam_manager",
                executable="omni_slam_manager",
                name="omni_slam_manager",
                output="screen",
                parameters=[
                    {
                        "map_root": map_root,
                        "map_source_path": map_source_path,
                    }
                ],
            ),
        ]
    )
