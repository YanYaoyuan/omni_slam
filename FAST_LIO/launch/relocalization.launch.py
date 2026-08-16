import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    Shutdown,
    TimerAction,
)
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def _validate_prior_map_path(raw_path):
  path = raw_path.strip()
  if not path:
    raise RuntimeError('map_path is required for relocalization')
  if not os.path.isabs(path):
    raise RuntimeError('map_path must be an absolute PCD path')
  normalized = os.path.realpath(path)
  if not normalized.lower().endswith('.pcd'):
    raise RuntimeError('map_path must reference a .pcd file')
  if not os.path.isfile(normalized):
    raise RuntimeError(f'prior PCD does not exist: {normalized}')
  if not os.access(normalized, os.R_OK):
    raise RuntimeError(f'prior PCD is not readable: {normalized}')
  if os.path.getsize(normalized) <= 0:
    raise RuntimeError(f'prior PCD is empty: {normalized}')
  return normalized


def _validate_relocalization_inputs(context):
  validated_map = _validate_prior_map_path(
      LaunchConfiguration('map_path').perform(context))
  return [SetLaunchConfiguration('map_path', validated_map)]


def _icp_exit_failed(returncode):
  """Treat ICP's intentional zero exit as success and every other exit as fatal."""
  return returncode != 0


def _shutdown_on_failed_icp(event, _context):
  if not _icp_exit_failed(event.returncode):
    return []
  return [Shutdown(reason=f'ICP relocalization failed with return code {event.returncode}')]


def generate_launch_description():

  map_path = LaunchConfiguration('map_path')
  declare_map_path = DeclareLaunchArgument(
      'map_path',
      default_value='',
      description='Required absolute PCD path shared by ICP and Fast-LIO')

  # icp relocalization
  map_odom_trans = Node(
      package='icp_relocalization',
      executable='transform_publisher',
      name='transform_publisher',
      output='screen',
      on_exit=Shutdown(reason='localization transform publisher exited')
  )

  icp_node = Node(
      package='icp_relocalization',
      executable='icp_node',
      name='icp_node',
      output='screen',
      parameters=[
          # --- Blue ---
          # {'initial_x':14.16},
          # {'initial_y':5.35},
          # {'initial_z':0.0},
          # {'initial_a':3.14},

          # --- Red ---
          {'initial_x':0.0},
          {'initial_y':0.0},
          {'initial_z':0.0},
          {'initial_a':0.0},

          {'map_voxel_leaf_size':0.5},
          {'cloud_voxel_leaf_size':0.3},
          {'map_frame_id':'map'},
          {'solver_max_iter':100},
          {'max_correspondence_distance':0.1},
          {'RANSAC_outlier_rejection_threshold':0.5},
          {'map_path':map_path},
          {'fitness_score_thre':0.2}, # 是最近点距离的平均值，越小越严格
          {'converged_count_thre':40}, # pcl pub at 20 hz, 2s
          {'pcl_type':'livox'},
      ],
      # icp_node deliberately exits with code 0 after publishing the initial
      # pose. Only startup/runtime failures terminate the localization epoch.
      on_exit=_shutdown_on_failed_icp,
  )
  
  # fast-lio localization   
  config = os.path.join(
      get_package_share_directory('fast_lio'), 'config', 'fast_lio_relocalization_param.yaml')
  
  fast_lio_node = Node(
      package='fast_lio',
      executable='fastlio_mapping',
      parameters=[
          config,
          {'prior_map_path':map_path},
      ],
      output='screen',
      remappings=[('/Odometry','/state_estimation')],
      on_exit=Shutdown(reason='Fast-LIO localization exited')
  )
        
  rviz_config_file = os.path.join(
    get_package_share_directory('fast_lio'), 'rviz', 'loam_livox.rviz')
  start_rviz = Node(
    package='rviz2',
    executable='rviz2',
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

  ld.add_action(declare_map_path)
  ld.add_action(OpaqueFunction(function=_validate_relocalization_inputs))
  ld.add_action(map_odom_trans)
  ld.add_action(start_rviz)
  ld.add_action(delayed_start_lio)

  return ld
