"""Semantic consumer only: never starts another manager or TF authority."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [DeclareLaunchArgument("weights"), DeclareLaunchArgument("output_directory"),
        DeclareLaunchArgument("ground_model", default_value=""),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("max_frames", default_value="300")]
    parameters = {k: LaunchConfiguration(k) for k in ("weights", "output_directory", "ground_model", "device")}
    parameters["use_sim_time"] = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    parameters["max_frames"] = ParameterValue(LaunchConfiguration("max_frames"), value_type=int)
    return LaunchDescription(arguments + [Node(package="omni_semantic_map",
        executable="semantic_trial", name="omni_semantic_trial", output="screen", parameters=[parameters])])
