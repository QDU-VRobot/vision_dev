# 项目知识库

**Generated:** 2026-03-06
**Commit:** 59d908d
**Branch:** rune_dev

## 概述

RoboMaster 自瞄视觉系统。ROS2 Humble 工作空间（C++14/17/20 + Python launch）。双管线架构：装甲板自瞄 (armor) + 能量机关 (rune)，通过 `mode` 参数在 launch 时切换。Pipeline: 相机采集 → 检测 (OpenCV + MLP/轮廓) → EKF 跟踪 → 弹道解算 → 串口指令至云台 MCU (LibXR)。

## 结构

```
vision_dev/
├── src/
│   ├── rm_auto_aim/              # 装甲板自瞄管线 (3 子包, 有 AGENTS.md)
│   │   ├── armor_detector/       # CV 检测 + PnP 解算 (有 AGENTS.md)
│   │   ├── armor_tracker/        # EKF 跟踪 + 弹道解算 (有 AGENTS.md)
│   │   └── auto_aim_interfaces/  # ROS2 msg: Armor, Target, Send, Velocity...
│   ├── rm_auto_aim_rune/         # 能量机关管线 (4 子包, 有 AGENTS.md)
│   │   ├── rm_rune_detector/     # Rune 检测 ROS2 节点
│   │   ├── rm_rune_tracker/      # Rune 跟踪 + 弹道解算
│   │   ├── rm_rune_interfaces/   # ROS2 msg: Rune, RuneTarget
│   │   └── rm_vision_core/       # 移植自华南虎 rm_vision_core 核心库
│   ├── rm_serial_driver/         # LibXR UART 桥接 (有 AGENTS.md)
│   ├── hik_camera/               # 海康工业相机驱动 (composable, 有 AGENTS.md)
│   ├── rm_vision/                # Launch 编排 + config 子模块
│   ├── rm_hand_eye_calibrate/    # 手眼标定工具 (LibXR + cv::calibrateHandEye)
│   ├── rm_gimbal_description/    # URDF xacro: gimbal_odom → yaw → pitch → camera
│   ├── rm_simulator_driver/      # 仿真替代 serial_driver (接收 Send, 发布 euler/fire)
│   └── libxr/                    # 嵌入式框架 (vendored 子模块, 有自己的 AGENTS.md)
├── tracker.sh                    # 看门狗脚本: 自动重启 tracker 节点
├── build/                        # colcon build 输出 (gitignored)
├── install/                      # colcon install 输出 (gitignored)
└── .clang-format                # Google base, 90 col, Allman braces
```

## 快速查找

| 任务 | 位置 | 备注 |
|------|------|------|
| 增改装甲板 ROS msg | `src/rm_auto_aim/auto_aim_interfaces/msg/` | 改后需重新构建 |
| 增改能量机关 ROS msg | `src/rm_auto_aim_rune/rm_rune_interfaces/msg/` | 改后需重新构建 |
| 能量机关检测参数 | `src/rm_auto_aim_rune/rm_rune_detector/config/rune_detector.yaml` | 单包可独立启动 |
| 能量机关测试管线 | `src/rm_auto_aim_rune/rm_rune_detector/launch/test_pipeline.launch.py` | 视频回放+模拟 TF |
| 机器人专属配置 | `src/rm_vision/rm_vision_bringup/config/` | 独立 git 仓库，按机器人类型切换分支 |
| 节点参数 | 各包 `config/` 目录下 YAML | 通过 launch 加载 |
| 相机内参 | `src/hik_camera/config/camera_info.yaml` | Hero 有 `camera_info_lob.yaml` |
| URDF / TF 树 | `src/rm_gimbal_description/urdf/rm_gimbal.urdf.xacro` | `odom2camera` xyz/rpy 来自 launch_params |
| Launch 编排 | `src/rm_vision/rm_vision_bringup/launch/` | `common.py` = 共享节点定义, `vision_bringup.launch.py` = 主入口 |
| 切换 armor/rune 管线 | `vision_bringup.launch.py` `mode` 参数 | `mode:=armor` (默认) 或 `mode:=rune` |
| CI/CD | `.github/workflows/build.yml` | 容器内 colcon build，SonarQube 扫描暂禁用 |
| LibXR 内部 | `src/libxr/libxr/` | 见其自己的 `AGENTS.md` |

## 数据流

### 装甲板管线 (mode=armor，默认)

```
HikCamera ──(/image_raw)──> ArmorDetector ──(/detector/armors)──> ArmorTracker
                                                                       │
                                       (/tracker/target + /tracker/send)│
                                                                       v
                             RMSerialDriver <──(LibXR UART)──> Gimbal MCU
                                  │
              (joint_state, /current_velocity, /lob_shot_switch) → TF + tracker
```

### 能量机关管线 (mode=rune)

```
HikCamera ──(/image_raw)──> RuneDetectorNode ──(/rune/target)──> RuneTrackerNode
                                  │                                     │
                        (/rune/debug_image)                    (/tracker/send)
                                                                       v
                             RMSerialDriver <──(LibXR UART)──> Gimbal MCU
```

关键 topic: `/image_raw`, `/detector/armors`, `/rune/target`, `/tracker/target`, `/tracker/send`, `/current_velocity`

## TF 树

```
gimbal_odom → [yaw_joint] → yaw_link → [pitch_joint] → pitch_link → [camera_joint] → camera_link → [fixed] → camera_optical_frame
```

- `yaw_joint`, `pitch_joint`: 由 serial_driver 发布的 JointState 驱动
- `camera_joint`: 固定偏移, 来自 `launch_params["odom2camera"]["xyz/rpy"]`

## 约定

- **命名空间**: 装甲板节点用 `rm_auto_aim`, 能量机关用 `rm_rune_detector` / `rm_auto_aim`
- **命名风格**: 类方法 PascalCase (`ImageCallback`, `DetectArmors`), 成员变量 snake_case 尾随下划线 (`tracker_`, `dt_`)
- **C++ 标准**: 混合 — hik_camera/rm_simulator_driver C++14；rm_serial_driver/rm_hand_eye_calibrate/rm_rune_tracker C++17；armor_detector/armor_tracker/rm_rune_detector C++20（以各包 CMakeLists 为准）
- **Composable 节点**: Camera + detector 运行在同一容器中；rune detector 既可独立启动也可 composable
- **Clang-format**: Google base, 90 列限制, Allman 花括号 (见 `.clang-format`)
- **文档语言**: 注释和 README 以中文为主
- **Config 子模块**: `rm_vision_bringup/config` 是独立 git 仓库，branch = 机器人类型

## 禁止事项

- **DO NOT** 在 LibXR 组件中释放内存 — 故意 allocate-once (见 `src/libxr/libxr/AGENTS.md`)
- **DO NOT** 在公共编译定义中添加 EIGEN_NO_IO — 在 `src/libxr/CMakeLists.txt` 中显式移除
- **DO NOT** 编辑 `src/libxr/libxr/` 下的文件 — vendored 子模块
- **DO NOT** 编辑 `build/` 或 `install/` — colcon 生成物
- **DO NOT** 在不改 MCU 固件的情况下修改 SharedTopic 传输格式
- Config 子模块 (`rm_vision_bringup/config/`) 设有 `ignore = all` — 变更独立跟踪

## 命令

```bash
# 构建
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

# 测试
colcon test --packages-select armor_detector
colcon test-result --verbose

# 运行 (完整系统 — 装甲板模式)
source install/setup.bash
ros2 launch rm_vision_bringup vision_bringup.launch.py robot:=<robot_type>

# 运行 (能量机关模式)
ros2 launch rm_vision_bringup vision_bringup.launch.py robot:=<robot_type> mode:=rune

# 运行 (无硬件 / 仿真)
ros2 launch rm_vision_bringup no_hardware.launch.py
ros2 launch rm_vision_bringup simulator.launch.py

# 运行 (单独启动)
ros2 launch hik_camera hik_camera.launch.py
ros2 launch rm_serial_driver ros2_libxr_launch.py
ros2 launch rm_hand_eye_calibrate hand_eye_calibrate.launch.py

# 相机标定
ros2 run camera_calibration cameracalibrator --size 11x8 --square 0.02 image:=/image_raw camera:=/hik_camera

# Git hooks (首次 clone)
git config core.hooksPath .githooks
```

## 注意事项

- Launch 系统在启动时对 config 子模块执行 `git checkout` (`vision_bringup.launch.py`)，分支必须存在
- Serial driver 延迟 1.5s, tracker 延迟 2.0s (等待 camera+detector 容器启动)
- Lob-shot 模式在运行时切换相机内参和弹道表 (仅 Hero 机器人)
- `tracker.sh` 依赖 konsole 且 bash 判断语句需有空格，使用前先检查
- Outpost 跟踪使用独立噪声参数 (`s2qxyz_outpost_` 等)
- Git hooks 在 checkout/merge 时自动更新子模块
- rm_auto_aim_rune 移植自华南理工华南虎战队 rm_vision_core，正在适配 ROS2
- simulator.launch.py 用 rm_simulator_driver 代替 serial_driver，无需硬件
