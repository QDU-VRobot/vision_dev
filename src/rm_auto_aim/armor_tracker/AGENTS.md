# armor_tracker

## OVERVIEW

Extended Kalman Filter target tracking with ballistic trajectory solving. Subscribes to detected armors, transforms to inertial frame via TF2, tracks robot state, computes fire solution (pitch/yaw + lead).

## STRUCTURE

```
armor_tracker/
├── include/armor_tracker/
│   ├── extended_kalman_filter.hpp  # EKF implementation (Eigen-based)
│   ├── tracker.hpp                 # State machine: DETECTING→TRACKING→TEMP_LOST→LOST
│   ├── tracker_node.hpp            # ROS2 node: subscriptions, publishers, parameters
│   └── SolveTrajectory.hpp         # Ballistic solver with gravity + air drag
├── src/
│   ├── extended_kalman_filter.cpp  # Predict/update with nonlinear observation model
│   ├── tracker.cpp                 # Target association (L2 distance), state transitions
│   ├── tracker_node.cpp            # ArmorsCallback, VelocityCallback, parameter loading
│   └── SolveTrajectory.cpp         # Iterative trajectory solve using precomputed tables
├── tools/
│   ├── TableGenerator.cpp          # Offline tool to generate trajectory lookup tables
│   └── *_table.bin                 # Precomputed tables per robot type + bullet speed
└── test/
    └── test_once_trajectory.cpp
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Tune EKF noise | `tracker_node.cpp` | `s2qxyz_`, `s2qyaw_`, `r_xyz_factor_`, `r_yaw_` from YAML |
| Outpost special params | `tracker_node.cpp` | `s2qxyz_outpost_`, `s2qyaw_outpost_`, `s2qr_outpost_` |
| Change state machine | `tracker.cpp` | `tracking_thres`, `lost_thres` control transitions |
| Modify ballistics | `SolveTrajectory.cpp` | Air drag model, iterative pitch solver |
| Generate new tables | `tools/TableGenerator.cpp` | Compile standalone, outputs `.bin` files |
| Change EKF state vector | `extended_kalman_filter.hpp/cpp` | State: [xc,yc,z,yaw,vxc,vyc,vz,vyaw,r] |

## CONVENTIONS

- **C++ standard**: C++17
- **State vector**: 9-dim — position of armor center (xc,yc,z), yaw, velocities, radius
- **Observation model**: Nonlinear — armor position = center + r*cos/sin(yaw)
- **Target association**: L2 Euclidean distance (SORT-inspired, single target)
- **Coordinate frame**: `odom` frame (gimbal center, IMU yaw-at-boot = X axis)
- Tracker selects armor closest to image center on init
- `robot_type` param controls which trajectory table to load

## ANTI-PATTERNS

- **DO NOT** change table format without updating `TableGenerator.cpp` — binary format is tightly coupled
- **DO NOT** assume constant dt — `dt_` is computed from message timestamps each frame
- Outpost tracking intentionally uses different noise params — do not unify with normal tracking
