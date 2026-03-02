# armor_detector

## OVERVIEW

OpenCV-based armor plate detection with MLP number classification and PnP 3D pose solving. Runs as composable node in same process as camera for zero-copy transport.

## STRUCTURE

```
armor_detector/
├── include/armor_detector/
│   ├── armor.hpp              # Armor/Light data structs
│   ├── detector.hpp           # Core detection pipeline
│   ├── detector_node.hpp      # ROS2 node wrapper
│   ├── number_classifier.hpp  # MLP classifier for armor numbers
│   └── pnp_solver.hpp         # cv::solvePnP wrapper (IPPE method)
├── src/
│   ├── detector.cpp           # preprocessImage → findLights → matchLights
│   ├── detector_node.cpp      # ImageCallback, parameter handling, debug publishers
│   ├── number_classifier.cpp  # Perspective warp → ROI → Otsu binarize → MLP inference
│   └── pnp_solver.cpp         # 3D position from 4 armor corners
├── model/
│   ├── mlp.onnx               # Trained MLP weights
│   └── label.txt              # Class labels for number classifier
└── test/
    ├── test_node_startup.cpp
    └── test_number_cls.cpp
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Tune binarization | `detector.cpp` `preprocessImage` | `binary_thres` param, grayscale-based |
| Change light filtering | `detector.cpp` `findLights` | min/max_ratio, max_angle |
| Change armor pairing | `detector.cpp` `matchLights` | center_distance, light_ratio thresholds |
| Swap classifier model | `model/mlp.onnx` + `label.txt` | Input: 20x28=560 flattened binary image |
| Change PnP method | `pnp_solver.cpp` | Currently `SOLVEPNP_IPPE` (coplanar) |

## CONVENTIONS

- **C++ standard**: C++14 (`-std=c++14` in CMakeLists.txt)
- **Strict warnings**: `-Wall -Wextra -Wpedantic -Werror` enabled
- Color detection uses R/B channel sum comparison, NOT HSV
- Light color judged by contour-interior pixel sums, not bounding rect
- `detect_color` is a dynamic parameter (0=red, 1=blue) — switchable at runtime

## ANTI-PATTERNS

- **DO NOT** use HSV for binarization — grayscale threshold is intentional due to industrial camera dynamic range limitations
- **DO NOT** change PnP method without understanding coplanarity assumption — armor corners are coplanar by definition
