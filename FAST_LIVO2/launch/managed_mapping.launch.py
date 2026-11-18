"""Manager-owned LIVO mapping; omni_tf_manager remains the only TF publisher."""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _node(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    output = Path(value("output_directory")).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LIVO output directory must be new or empty")
    return [
        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="omni_fast_livo",
            parameters=[
                value("config"),
                {
                    "use_sim_time": value("use_sim_time").lower() == "true",
                    "common.odom_frame": value("odom_frame"),
                    "common.tracking_frame": value("tracking_frame"),
                    "common.img_en": 1,
                    "vio.lidar_timed_fusion": True,
                    "runtime.keyframe_archive": True,
                    "publish.tf_en": False,
                    "runtime.output_directory": str(output),
                    "runtime.map_file_path": value("map_path"),
                },
            ],
            output="screen",
            remappings=[
                ("Odometry", "/state_estimation"),
                ("cloud_registered", "/cloud_registered_global"),
                ("keyframes", "/omni/slam/livo/keyframes"),
                ("semantic_frames", "/omni/slam/livo/semantic_frames"),
                ("rgb_img", "/omni/slam/livo/image"),
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config"),
            DeclareLaunchArgument("output_directory"),
            DeclareLaunchArgument("map_path"),
            DeclareLaunchArgument("odom_frame", default_value="omni_odom"),
            DeclareLaunchArgument("tracking_frame", default_value="omni_imu_link"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            OpaqueFunction(function=_node),
        ]
    )
