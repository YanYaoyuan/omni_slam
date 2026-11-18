from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    names=['vocabulary_directory','feature_space','output_directory']
    return LaunchDescription([DeclareLaunchArgument(n) for n in names]+[
        Node(package='omni_semantic_map',executable='semantic_categories',output='screen',
             parameters=[{n:LaunchConfiguration(n) for n in names}])])
