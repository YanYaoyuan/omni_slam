import os.path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_file = os.path.join(
        package_path, 'config', 'vbot.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    map_path = LaunchConfiguration('map_path')
    map_frame = LaunchConfiguration('map_frame')
    odom_frame = LaunchConfiguration('odom_frame')
    base_frame = LaunchConfiguration('base_frame')
    tracking_frame = LaunchConfiguration('tracking_frame')
    lidar_frame = LaunchConfiguration('lidar_frame')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true')
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Yaml config file path')
    declare_map_path_cmd = DeclareLaunchArgument(
        'map_path',
        default_value='/tmp/omni_slam_work/map.pcd',
        description='Absolute output PCD path used by the map_save service')
    declare_odom_frame_cmd = DeclareLaunchArgument(
        'odom_frame', default_value='omni_odom')
    declare_map_frame_cmd = DeclareLaunchArgument(
        'map_frame', default_value='omni_map')
    declare_base_frame_cmd = DeclareLaunchArgument(
        'base_frame', default_value='omni_base_link')
    declare_tracking_frame_cmd = DeclareLaunchArgument(
        'tracking_frame', default_value='omni_imu_link')
    declare_lidar_frame_cmd = DeclareLaunchArgument(
        'lidar_frame', default_value='omni_lidar_link')

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='omni_fast_lio',
        parameters=[
            config_file,
            {
                'use_sim_time': use_sim_time,
                'common.map_frame_id': map_frame,
                'common.odom_frame_id': odom_frame,
                'common.sensor_frame_id': tracking_frame,
                'common.lidar_frame_id': lidar_frame,
                'common.base_frame_id': base_frame,
                'common.send_odom_base_tf': False,
                'publish.tf_en': False,
                'map_file_path': map_path,
            },
        ],
        output='screen',
        remappings=[
            ('/Odometry', '/state_estimation'),
            ('/cloud_registered', '/cloud_registered_global'),
        ])

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_map_path_cmd)
    ld.add_action(declare_odom_frame_cmd)
    ld.add_action(declare_map_frame_cmd)
    ld.add_action(declare_base_frame_cmd)
    ld.add_action(declare_tracking_frame_cmd)
    ld.add_action(declare_lidar_frame_cmd)
    ld.add_action(fast_lio_node)
    return ld
