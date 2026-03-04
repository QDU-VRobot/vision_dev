cmake_minimum_required(VERSION 3.8)
project(rm_rune_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================== 导出 compile_commands.json ====================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==================== 依赖查找 ====================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rm_rune_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# ==================== rm_vision_core 依赖 ====================
find_package(rm_vision_core REQUIRED)

# ==================== 公共依赖列表 ====================
set(DEPS
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_geometry_msgs
  rm_rune_interfaces
  rm_vision_core
)

# ==================== 主节点组件库 ====================
add_library(${PROJECT_NAME} SHARED
  src/rune_detector_node.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME} ${DEPS})

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  rm_vision_core::rm_vision_core
)

# 注册组件
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "rm_rune_detector::RuneDetectorNode"
  EXECUTABLE rune_detector_node
)

# ==================== 测试: 单张图片检测 ====================
add_executable(test_image_node test/test_image_node.cpp)
target_include_directories(test_image_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(test_image_node rclcpp sensor_msgs cv_bridge rm_vision_core)
target_link_libraries(test_image_node ${OpenCV_LIBS} rm_vision_core::rm_vision_core)

# ==================== 测试: 视频检测追踪 ====================
add_executable(test_video_node test/test_video_node.cpp)
target_include_directories(test_video_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(test_video_node rclcpp sensor_msgs cv_bridge rm_vision_core)
target_link_libraries(test_video_node ${OpenCV_LIBS} rm_vision_core::rm_vision_core)

# ==================== 测试: 模拟 TF 发布 ====================
add_executable(test_mock_tf_node test/test_mock_tf_node.cpp)
ament_target_dependencies(test_mock_tf_node
  rclcpp sensor_msgs tf2 tf2_ros geometry_msgs)

# ==================== 测试: 视频帧发布器 ====================
add_executable(test_video_publisher_node test/test_video_publisher_node.cpp)
target_include_directories(test_video_publisher_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(test_video_publisher_node
  rclcpp sensor_msgs cv_bridge)
target_link_libraries(test_video_publisher_node ${OpenCV_LIBS})

# ==================== 安装 ====================
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  test_image_node
  test_video_node
  test_mock_tf_node
  test_video_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# ==================== 导出 ====================
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(${DEPS})

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
