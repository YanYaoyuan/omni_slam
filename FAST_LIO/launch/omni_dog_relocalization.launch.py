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
    max_correspondence_distance = LaunchConfiguration('max_correspondence_distance')
    fitness_score_thre = LaunchConfiguration('fitness_score_thre')
    converged_count_thre = LaunchConfiguration('converged_count_thre')
    map_voxel_leaf_size = LaunchConfiguration('map_voxel_leaf_size')
    cloud_voxel_leaf_size = LaunchConfiguration('cloud_voxel_leaf_size')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')

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
    declare_max_correspondence_distance = DeclareLaunchArgument(
        'max_correspondence_distance',
        default_value='1.0',
        description='ICP max correspondence distance in meters')
    declare_fitness_score_thre = DeclareLaunchArgument(
        'fitness_score_thre',
        default_value='0.2',
        description='ICP publishes initial pose only when fitness is below this value')
    declare_converged_count_thre = DeclareLaunchArgument(
        'converged_count_thre',
        default_value='10',
        description='Number of consecutive good ICP results required before publishing pose')
    declare_map_voxel_leaf_size = DeclareLaunchArgument(
        'map_voxel_leaf_size',
        default_value='0.5',
        description='Voxel leaf size for the prior map')
    declare_cloud_voxel_leaf_size = DeclareLaunchArgument(
        'cloud_voxel_leaf_size',
        default_value='0.2',
        description='Voxel leaf size for the live scan')
    declare_pointcloud_topic = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/front_lidar',
        description='PointCloud2 topic used by ICP')

    map_odom_trans = Node(
        package='icp_relocalization',
        executable='transform_publisher',
        name='transform_publisher',
        output='screen',
        parameters=[
            {'map_frame_id': 'lio_map'},
            {'odom_frame_id': 'lio_odom'},
        ])

    global_odometry_node = Node(
        package='icp_relocalization',
        executable='global_odometry_publisher',
        name='global_odometry_publisher',
        output='screen',
        parameters=[
            {'input_topic': '/state_estimation'},
            {'output_topic': '/state_estimation_global'},
            {'target_frame': 'lio_map'},
        ])

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
            {'map_voxel_leaf_size': ParameterValue(map_voxel_leaf_size, value_type=float)},
            {'cloud_voxel_leaf_size': ParameterValue(cloud_voxel_leaf_size, value_type=float)},
            {'map_frame_id': 'lio_map'},
            {'solver_max_iter': 100},
            {'max_correspondence_distance': ParameterValue(max_correspondence_distance, value_type=float)},
            {'RANSAC_outlier_rejection_threshold': 0.5},
            {'map_path': map_path},
            {'fitness_score_thre': ParameterValue(fitness_score_thre, value_type=float)},
            {'converged_count_thre': ParameterValue(converged_count_thre, value_type=int)},
            {'pcl_type': 'pointcloud2'},
            {'pointcloud_topic': pointcloud_topic},
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
    ld.add_action(declare_max_correspondence_distance)
    ld.add_action(declare_fitness_score_thre)
    ld.add_action(declare_converged_count_thre)
    ld.add_action(declare_map_voxel_leaf_size)
    ld.add_action(declare_cloud_voxel_leaf_size)
    ld.add_action(declare_pointcloud_topic)
    ld.add_action(map_odom_trans)
    ld.add_action(global_odometry_node)
    ld.add_action(delayed_start_lio)
    return ld
