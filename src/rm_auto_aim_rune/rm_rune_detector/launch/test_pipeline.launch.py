"""
能量机关完整管线测试 Launch 文件

启动:
  ros2 launch rm_rune_detector test_pipeline.launch.py video_path:=/path/to/video.mp4

功能:
  1. test_mock_tf_node: 发布模拟的 TF 变换树和 CameraInfo
  2. test_video_publisher_node: 将视频逐帧发布为 /image_raw 话题
  3. rune_detector_node: 订阅图像并执行检测，发布 /rune/target + /rune/markers + /rune/debug_image

可选启动 rviz2 可视化:
  rviz2 -d $(ros2 pkg prefix rm_rune_detector)/share/rm_rune_detector/rviz/rune_detector.rviz
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

    video_path_arg = DeclareLaunchArgument(
        'video_path', default_value='',
        description='Path to the test video file')

    detect_color_arg = DeclareLaunchArgument(
        'detect_color', default_value='0',
        description='Detection color (0=RED, 1=BLUE)')

    color_threshold_arg = DeclareLaunchArgument(
        'color_threshold', default_value='100',
        description='Color threshold for binarization')

    fx_arg = DeclareLaunchArgument('fx', default_value='1280.0')
    fy_arg = DeclareLaunchArgument('fy', default_value='1280.0')
    cx_arg = DeclareLaunchArgument('cx', default_value='640.0')
    cy_arg = DeclareLaunchArgument('cy', default_value='512.0')
    fps_arg = DeclareLaunchArgument('fps', default_value='30.0')

    mock_tf_node = Node(
        package='rm_rune_detector',
        executable='test_mock_tf_node',
        name='mock_tf_node',
        parameters=[{
            'yaw': 0.0,
            'pitch': 0.0,
            'fx': LaunchConfiguration('fx'),
            'fy': LaunchConfiguration('fy'),
            'cx': LaunchConfiguration('cx'),
            'cy': LaunchConfiguration('cy'),
            'publish_camera_info': False,  # video_publisher 会发布
        }],
        output='screen',
    )

    video_pub_node = Node(
        package='rm_rune_detector',
        executable='test_video_publisher_node',
        name='video_publisher_node',
        parameters=[{
            'video_path': LaunchConfiguration('video_path'),
            'fx': LaunchConfiguration('fx'),
            'fy': LaunchConfiguration('fy'),
            'cx': LaunchConfiguration('cx'),
            'cy': LaunchConfiguration('cy'),
            'publish_rate': LaunchConfiguration('fps'),
            'loop': True,
        }],
        output='screen',
    )

    detector_node = Node(
        package='rm_rune_detector',
        executable='rune_detector_node',
        name='rune_detector_node',
        parameters=[
            default_params_file,
            {
                'debug': True,
                'detect_color': LaunchConfiguration('detect_color'),
                'color_threshold': LaunchConfiguration('color_threshold'),
            },
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        video_path_arg,
        detect_color_arg,
        color_threshold_arg,
        fx_arg, fy_arg, cx_arg, cy_arg, fps_arg,
        mock_tf_node,
        video_pub_node,
        detector_node,
    ])
