from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_managed_stack(context):
    robot_profile = LaunchConfiguration("robot_profile").perform(context)
    requested_tf_profile = LaunchConfiguration("tf_profile_path").perform(context)
    tf_manager_share = Path(get_package_share_directory("omni_tf_manager"))
    default_profiles = {
        "omni_dog": tf_manager_share / "config" / "omni_dog.yaml",
        "vbot": tf_manager_share / "config" / "omni_vbot_dog.yaml",
    }
    if robot_profile not in default_profiles:
        raise RuntimeError(f"unsupported robot_profile: {robot_profile}")
    tf_profile_path = requested_tf_profile or str(default_profiles[robot_profile])
    tf_manager_launch = tf_manager_share / "launch" / "omni_tf_manager.launch.py"

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(tf_manager_launch)),
            launch_arguments={
                "config_file": tf_profile_path,
                "mode": "profile",
            }.items(),
            condition=IfCondition(LaunchConfiguration("start_tf_manager")),
        ),
        Node(
            package="omni_slam_manager",
            executable="omni_slam_manager",
            name="omni_slam_manager",
            output="screen",
            parameters=[
                {
                    "robot_profile": robot_profile,
                    "map_root": LaunchConfiguration("map_root"),
                    "map_source_path": LaunchConfiguration("map_source_path"),
                    "tf_profile_path": tf_profile_path,
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_profile", default_value="omni_dog"),
            DeclareLaunchArgument("map_root", default_value="/var/lib/omni/maps"),
            DeclareLaunchArgument(
                "map_source_path", default_value="/tmp/omni_slam_work/map.pcd"
            ),
            DeclareLaunchArgument(
                "tf_profile_path",
                default_value="",
                description=(
                    "Optional TF profile override; empty selects the matching "
                    "omni_dog/omni_vbot_dog profile"
                ),
            ),
            DeclareLaunchArgument("start_tf_manager", default_value="true"),
            OpaqueFunction(function=_launch_managed_stack),
        ]
    )
