import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node


sys.path.append(
    os.path.join(get_package_share_directory("rm_vision_bringup"), "launch")
)


def _build_after_checkout(context, *args, **kwargs):
    """
    在 checkout 完成后执行
    """
    from common import (
        node_params,
        launch_params,
        robot_state_publisher,
        get_tracker_node,
        get_rune_tracker_node,
    )

    robot_type = LaunchConfiguration("robot").perform(context)

    def get_camera_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name="camera_node",
            parameters=[node_params, {"robot_type": robot_type}],
            extra_arguments=[{"use_intra_process_comms": True}],
        )

    def get_camera_detector_container(camera_node, detector_node, log_key, log_name):
        return ComposableNodeContainer(
            name="camera_detector_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                camera_node,
                detector_node,
            ],
            output="both",
            emulate_tty=True,
            ros_arguments=[
                "--ros-args",
                "--log-level",
                log_name + ":=" + launch_params[log_key],
            ],
            on_exit=Shutdown(),
        )

    hik_camera_node = get_camera_node("hik_camera", "HikCamera::HikCameraNode")
    mv_camera_node = get_camera_node(
        "mindvision_camera", "mindvision_camera::MVCameraNode"
    )

    if launch_params["camera"] == "hik":
        camera_node = hik_camera_node
    elif launch_params["camera"] == "mv":
        camera_node = mv_camera_node
    else:
        raise RuntimeError(f"Unknown camera type: {launch_params['camera']}")

    from launch.actions import TimerAction

    # 根据 mode 选择 detector/tracker pipeline
    mode = LaunchConfiguration("mode").perform(context)

    if mode == "rune":
        detector_composable = ComposableNode(
            package="rm_rune_detector",
            plugin="rm_rune_detector::RuneDetectorNode",
            name="rune_detector_node",
            parameters=[node_params, {"robot_type": robot_type}],
            extra_arguments=[{"use_intra_process_comms": True}],
        )
        cam_detector = get_camera_detector_container(
            camera_node,
            detector_composable,
            "rune_detector_log_level",
            "rune_detector_node",
        )
        delay_tracker_node = TimerAction(
            period=2.0, actions=[get_rune_tracker_node(robot_type)]
        )
    else:
        detector_composable = ComposableNode(
            package="armor_detector",
            plugin="rm_auto_aim::ArmorDetectorNode",
            name="armor_detector",
            parameters=[node_params, {"robot_type": robot_type}],
            extra_arguments=[{"use_intra_process_comms": True}],
        )
        cam_detector = get_camera_detector_container(
            camera_node,
            detector_composable,
            "detector_log_level",
            "armor_detector",
        )
        delay_tracker_node = TimerAction(
            period=2.0, actions=[get_tracker_node(robot_type)]
        )

    serial_driver_node = Node(
        package="rm_serial_driver",
        executable="rm_serial_driver_node",
        name="serial_driver",
        output="both",
        emulate_tty=True,
        parameters=[node_params, {"robot_type": robot_type}],
        on_exit=Shutdown(),
        ros_arguments=[
            "--ros-args",
            "--log-level",
            "serial_driver:=" + launch_params["serial_log_level"],
        ],
    )

    delay_serial_node = TimerAction(period=1.5, actions=[serial_driver_node])

    return [
        robot_state_publisher,
        cam_detector,
        delay_serial_node,
        delay_tracker_node,
    ]


def generate_launch_description():
    ws_root = LaunchConfiguration("ws_root")
    robot = LaunchConfiguration("robot")

    config_repo_rel = "src/rm_vision/rm_vision_bringup/config"

    checkout_robot = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            "set -e; "
            'cd "$WS_ROOT"/' + config_repo_rel + "; "
            "git rev-parse --is-inside-work-tree >/dev/null 2>&1; "
            'git checkout "$BRANCH"; '
            "git status --porcelain",
        ],
        additional_env={
            "WS_ROOT": ws_root,
            "BRANCH": robot,
        },
        output="screen",
    )

    build_nodes = OpaqueFunction(function=_build_after_checkout)

    return LaunchDescription(
        [
            DeclareLaunchArgument("ws_root", default_value=os.getcwd()),
            DeclareLaunchArgument("robot", default_value="main"),
            DeclareLaunchArgument(
                "mode",
                default_value="armor",
                description="Pipeline mode: armor or rune",
            ),
            checkout_robot,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=checkout_robot,
                    on_exit=[build_nodes],
                )
            ),
        ]
    )
