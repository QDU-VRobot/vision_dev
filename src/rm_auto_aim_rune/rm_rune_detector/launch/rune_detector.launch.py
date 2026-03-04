"""
能量机关检测节点 Launch 文件

启动: ros2 launch rm_rune_detector rune_detector.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rm_rune_detector')
    default_params_file = os.path.join(pkg_share, 'config', 'rune_detector.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the rune detector parameters file'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Enable debug image and RViz marker output'
    )

    rune_detector_node = Node(
        package='rm_rune_detector',
        executable='rune_detector_node',
        name='rune_detector_node',
        parameters=[
            LaunchConfiguration('params_file'),
            {'debug': LaunchConfiguration('debug')},
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        params_file_arg,
        debug_arg,
        rune_detector_node,
    ])
