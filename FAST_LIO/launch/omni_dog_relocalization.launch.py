import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    fast_lio_share = get_package_share_directory('fast_lio')
    config_file = LaunchConfiguration('config_file')
    map_path = LaunchConfiguration('map_path')
    initial_x = LaunchConfiguration('initial_x')
    initial_y = LaunchConfiguration('initial_y')
    initial_z = LaunchConfiguration('initial_z')
    initial_yaw = LaunchConfiguration('initial_yaw')

    default_config_file = os.path.join(
        fast_lio_share, 'config', 'omni_dog_relocalization.yaml')

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Fast-LIO relocalization parameter file')
    declare_map_path = DeclareLaunchArgument(
        'map_path',
        default_value='',
        description='Prior PCD map path used by both ICP and Fast-LIO')
    declare_initial_x = DeclareLaunchArgument('initial_x', default_value='0.0')
    declare_initial_y = DeclareLaunchArgument('initial_y', default_value='0.0')
    declare_initial_z = DeclareLaunchArgument('initial_z', default_value='0.0')
    declare_initial_yaw = DeclareLaunchArgument('initial_yaw', default_value='0.0')

    map_odom_trans = Node(
        package='icp_relocalization',
        executable='transform_publisher',
        name='transform_publisher',
        output='screen')

    icp_node = Node(
        package='icp_relocalization',
        executable='icp_node',
        name='icp_node',
        output='screen',
        parameters=[
            {'initial_x': ParameterValue(initial_x, value_type=float)},
            {'initial_y': ParameterValue(initial_y, value_type=float)},
            {'initial_z': ParameterValue(initial_z, value_type=float)},
            {'initial_a': ParameterValue(initial_yaw, value_type=float)},
            {'map_voxel_leaf_size': 0.5},
            {'cloud_voxel_leaf_size': 0.3},
            {'map_frame_id': 'map'},
            {'solver_max_iter': 100},
            {'max_correspondence_distance': 0.1},
            {'RANSAC_outlier_rejection_threshold': 0.5},
            {'map_path': map_path},
            {'fitness_score_thre': 0.2},
            {'converged_count_thre': 40},
            {'pcl_type': 'pointcloud2'},
            {'pointcloud_topic': '/front_lidar'},
        ])

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[
            config_file,
            {'prior_map_path': map_path},
        ],
        output='screen',
        remappings=[('/Odometry', '/state_estimation')])

    delayed_start_lio = TimerAction(
        period=5.0,
        actions=[
            icp_node,
            fast_lio_node,
        ])

    ld = LaunchDescription()
    ld.add_action(declare_config_file)
    ld.add_action(declare_map_path)
    ld.add_action(declare_initial_x)
    ld.add_action(declare_initial_y)
    ld.add_action(declare_initial_z)
    ld.add_action(declare_initial_yaw)
    ld.add_action(map_odom_trans)
    ld.add_action(delayed_start_lio)
    return ld
