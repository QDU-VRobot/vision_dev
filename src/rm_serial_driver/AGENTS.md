# rm_serial_driver

## OVERVIEW

UART bridge between ROS2 and gimbal MCU using LibXR's SharedTopic protocol. Receives IMU quaternion + bullet speed from MCU, publishes as ROS2 topics. Receives fire commands from tracker, sends as LibXR topic to MCU.

## STRUCTURE

```
rm_serial_driver/
├── include/rm_serial_driver/
│   ├── rm_serial_driver.hpp    # ROS2 node + LibXR init
│   ├── SharedTopic.hpp         # LibXR UART server (MCU→PC data parsing)
│   └── SharedTopicClient.hpp   # LibXR UART client (PC→MCU data sending)
├── src/
│   └── rm_serial_driver.cpp    # Node implementation, callbacks, topic bridging
├── config/
│   └── serial.yaml             # UART device path, baud rate
├── launch/
│   └── ros2_libxr_launch.py
└── test/
    ├── test_gimbal_controll.cpp      # Gimbal control test
    └── test_timestamp_calibrator.cpp # Timestamp sync test
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change UART config | `config/serial.yaml` | Device path, baud rate |
| Add new MCU topic | `rm_serial_driver.hpp` `XRobotMain()` | Register in SharedTopic/SharedTopicClient |
| Modify data bridging | `rm_serial_driver.cpp` | ROS2 ↔ LibXR topic mapping |
| Timestamp offset | `rm_serial_driver.cpp` | `timestamp_offset_` param for sensor sync |

## CONVENTIONS

- LibXR topics initialized in static `XRobotMain()` function — runs before ROS2 node constructor
- MCU→PC topics: `ahrs_quaternion`, `lob_shot` (via SharedTopic server)
- PC→MCU topics: `target_euler`, `fire_notify` (via SharedTopicClient)
- `timestamp_offset_` compensates for MCU-camera time drift
- Hero robot (`is_hero_`) gets special lob-shot handling

## ANTI-PATTERNS

- **DO NOT** add dynamic memory allocation in LibXR components — allocate-once pattern required
- **DO NOT** change SharedTopic wire format without matching MCU firmware
