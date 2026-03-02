# rm_auto_aim

## OVERVIEW

Core auto-aim pipeline: 3 ROS2 packages forming detect → track → solve chain. Forked from [chenjunnn/rm_auto_aim](https://github.com/chenjunnn/rm_auto_aim) with significant extensions (trajectory solver, outpost tracking, lob-shot).

## STRUCTURE

```
rm_auto_aim/
├── armor_detector/        # Image → armor 3D positions (has own AGENTS.md)
├── armor_tracker/         # Armor positions → tracked target + fire command (has own AGENTS.md)
└── auto_aim_interfaces/   # ROS2 msg definitions shared across packages
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new message type | `auto_aim_interfaces/msg/` | Add to `CMakeLists.txt` too |
| Change detection algo | `armor_detector/src/detector.cpp` | Preprocessing, light finding, matching |
| Tune EKF parameters | `armor_tracker/src/tracker_node.cpp` | `s2q*` and `r_*` params from YAML |
| Modify ballistics | `armor_tracker/src/SolveTrajectory.cpp` | Uses precomputed `.bin` lookup tables |
| Change MLP model | `armor_detector/model/mlp.onnx` | Labels in `model/label.txt` |

## MESSAGE TYPES

| Message | Purpose | Key fields |
|---------|---------|------------|
| `Armors` | Detector → Tracker | header + `Armor[]` array |
| `Armor` | Single detection | number, type, pose, distance_to_center |
| `Target` | Tracker output | position, velocity, yaw, v_yaw, radius, aiming_point |
| `Send` | Fire command → serial | pitch, yaw, is_fire, idx |
| `Velocity` | Bullet speed from MCU | velocity float |
| `TrackerInfo` | Debug: prediction vs measurement | position_diff, yaw_diff |

## CONVENTIONS

- All nodes live in `rm_auto_aim` namespace
- Composable node plugins registered via `rclcpp_components` — detector runs in-process with camera
- Debug publishers gated by dynamic `debug` parameter
- `robot_type` parameter selects per-robot behavior at runtime
