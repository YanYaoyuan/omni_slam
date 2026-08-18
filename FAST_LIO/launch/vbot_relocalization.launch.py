import os.path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    fast_lio_share = get_package_share_directory('fast_lio')
    vbot_config = os.path.join(
        fast_lio_share, 'config', 'vbot_relocalization.yaml')

    argument_defaults = (
        ('map_path', '', 'Required absolute PCD map path'),
        ('initial_x', '0.0', 'Initial X in the map frame, in meters'),
        ('initial_y', '0.0', 'Initial Y in the map frame, in meters'),
        ('initial_z', '0.0', 'Initial Z in the map frame, in meters'),
        ('initial_yaw', '0.0', 'Initial yaw in the map frame, in radians'),
        ('max_correspondence_distance', '1.0', 'ICP maximum correspondence distance'),
        ('fitness_score_thre', '0.2', 'Maximum accepted ICP fitness score'),
        ('converged_count_thre', '10', 'Required consecutive accepted ICP results'),
        ('map_voxel_leaf_size', '0.5', 'Prior-map voxel size'),
        ('cloud_voxel_leaf_size', '0.2', 'Live-scan voxel size'),
        ('config_file', vbot_config, 'Vbot Fast-LIO relocalization parameters'),
        ('pointcloud_topic', '/lidar_points', 'Vbot PointCloud2 topic used by ICP'),
        ('sensor_frame', 'lidar_frame', 'Vbot LiDAR frame'),
    )

    forwarded_arguments = {
        name: LaunchConfiguration(name)
        for name, _default, _description in argument_defaults
    }
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            fast_lio_share, 'launch', 'omni_dog_relocalization.launch.py')),
        launch_arguments=forwarded_arguments.items())

    description = LaunchDescription()
    for name, default, help_text in argument_defaults:
        description.add_action(DeclareLaunchArgument(
            name, default_value=default, description=help_text))
    description.add_action(localization)
    return description
