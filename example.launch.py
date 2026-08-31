import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():

  config_path = os.path.join(
      get_package_share_directory('fast_lio'), 'config') 

  # TF is owned by omni_tf_manager. This example only starts pose producers.
  icp_node = Node(
      package='icp_relocalization',
      executable='icp_node',
      name='omni_icp_relocalization',
      output='screen',
      parameters=[
          {'initial_x':0.0},
          {'initial_y':0.0},
          {'initial_z':0.0},
          {'initial_a':0.0},

          {'map_voxel_leaf_size':0.5},
          {'cloud_voxel_leaf_size':0.3},
          {'map_frame_id':'omni_map'},
          {'solver_max_iter':100},
          {'max_correspondence_distance':0.1},
          {'RANSAC_outlier_rejection_threshold':0.5},
          # {'map_path':'/home/sentry_ws/src/sentry_bringup/maps/CC#0.pcd'},
          {'map_path':'/home/sentry_ws/src/FAST_LIO_SAM/PCD/GlobalMap.pcd'},
          {'fitness_score_thre':0.2}, # 是最近点距离的平均值，越小越严格
          {'converged_count_thre':40}, # pcl pub at 20 hz, 2s
          {'pcl_type':'livox'},
      ],
  )
  
  # fast-lio localization   
  fast_lio_param = os.path.join(
      config_path, 'fast_lio_relocalization_param.yaml')
  fast_lio_node = Node(
      package='fast_lio',
      executable='fastlio_mapping',
      name='omni_fast_lio',
      parameters=[
          fast_lio_param,
          {
              'common.odom_frame_id': 'omni_odom',
              'common.sensor_frame_id': 'omni_imu_link',
              'common.base_frame_id': 'omni_base_link',
              'common.send_odom_base_tf': False,
              'publish.tf_en': False,
          },
      ],
      output='screen',
      remappings=[('/Odometry','/state_estimation')]
  )
        
  rviz_config_file = os.path.join(
    get_package_share_directory('icp_relocalization'), 'rviz', 'loam_livox.rviz')
  start_rviz = Node(
    package='rviz2',
    executable='rviz2',
    name='omni_slam_rviz',
    arguments=['-d', rviz_config_file,'--ros-args', '--log-level', 'warn'],
    output='screen'
  )

  delayed_start_lio = TimerAction(
    period=5.0,
    actions=[
      icp_node,
      fast_lio_node
    ]
  )

  ld = LaunchDescription()

  ld.add_action(start_rviz)
  ld.add_action(delayed_start_lio)

  return ld
