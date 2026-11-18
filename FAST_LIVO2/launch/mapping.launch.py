"""Run the native Humble port with a verified sensor parameter file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config", description="ROS 2 YAML with sensor calibration"
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="fast_livo",
                executable="fastlivo_mapping",
                name="fast_livo",
                parameters=[
                    LaunchConfiguration("config"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
                output="screen",
            ),
        ]
    )
