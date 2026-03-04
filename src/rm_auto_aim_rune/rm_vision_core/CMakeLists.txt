cmake_minimum_required(VERSION 3.10)
project(rm_vision_core)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# ===========================
# ROS2 依赖查找
# ===========================
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)

# ===========================
# 自定义模块宏（适配 ROS2）
# ===========================
set(VC_MODULE_LIST "") # 用于记录所有模块

macro(viscore_add_module MODULE_NAME)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(options INTERFACE)
  set(oneValueArgs "")
  set(multiValueArgs DEPENDS EXTERNAL)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # 构建目标名称
  set(TARGET_NAME ${PROJECT_NAME}_${MODULE_NAME})

  # 收集源文件
  file(GLOB_RECURSE MODULE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

  if(ARG_INTERFACE)
    # INTERFACE 库（仅头文件）
    add_library(${TARGET_NAME} INTERFACE)
    target_include_directories(
      ${TARGET_NAME}
      INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)

    # 处理依赖
    foreach(dep ${ARG_DEPENDS})
      target_link_libraries(${TARGET_NAME} INTERFACE ${PROJECT_NAME}_${dep})
    endforeach()

    foreach(ext_dep ${ARG_EXTERNAL})
      target_link_libraries(${TARGET_NAME} INTERFACE ${ext_dep})
    endforeach()

    # INTERFACE 库需要导出
    install(TARGETS ${TARGET_NAME} EXPORT ${PROJECT_NAME}_Targets)

  else()
    # 普通库
    add_library(${TARGET_NAME} ${MODULE_SOURCES})

    target_include_directories(
      ${TARGET_NAME}
      PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
             $<INSTALL_INTERFACE:include>
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # 处理依赖
    foreach(dep ${ARG_DEPENDS})
      target_link_libraries(${TARGET_NAME} PUBLIC ${PROJECT_NAME}_${dep})
    endforeach()

    foreach(ext_dep ${ARG_EXTERNAL})
      target_link_libraries(${TARGET_NAME} PUBLIC ${ext_dep})
    endforeach()

    # 安装库文件
    install(
      TARGETS ${TARGET_NAME}
      EXPORT ${PROJECT_NAME}_Targets
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin)
  endif()

  # 安装头文件
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/vc")
    install(
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vc
      DESTINATION include
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp"
      PATTERN "*.yml"
      PATTERN "*.yaml")
  endif()

  # 记录模块
  list(APPEND VC_MODULE_LIST ${TARGET_NAME})
  set(VC_MODULE_LIST
      ${VC_MODULE_LIST}
      PARENT_SCOPE)

  message(STATUS "✓ ROS2 Module: ${TARGET_NAME}")
endmacro()

# ===========================
# 添加子目录（按依赖顺序）
# ===========================
add_subdirectory(common/core)
add_subdirectory(common/camera)
add_subdirectory(common/contour_proc)
add_subdirectory(common/dataio)
add_subdirectory(common/math)
add_subdirectory(common/modules/detector)
add_subdirectory(common/modules/feature)

# ★ 添加 modules 目录（如果包含库模块）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/modules")
  add_subdirectory(modules)
endif()
# ===========================
# 创建统一聚合库
# ===========================
add_library(${PROJECT_NAME} INTERFACE)
# 链接所有子模块
foreach(module ${VC_MODULE_LIST})
  target_link_libraries(${PROJECT_NAME} INTERFACE ${module})
endforeach()
# 聚合库导出
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_Targets)
message(
  STATUS "✓ Aggregated Library: ${PROJECT_NAME} (includes ${VC_MODULE_LIST})")
# ===========================
# 安装 YML 配置文件
# ===========================
# 查找并安装所有 .yml 文件
file(GLOB_RECURSE YML_FILES
     "${CMAKE_CURRENT_SOURCE_DIR}/common/camera/include/vc/camera/yml/*.yml"
     "${CMAKE_CURRENT_SOURCE_DIR}/common/camera/include/vc/camera/yml/*.yaml")
if(YML_FILES)
  # 安装到 share 目录
  install(
    DIRECTORY common/camera/include/vc/camera/yml
    DESTINATION share/${PROJECT_NAME}/config
    FILES_MATCHING
    PATTERN "*.yml"
    PATTERN "*.yaml")
  message(
    STATUS "✓ Installing YML configs to: share/${PROJECT_NAME}/config/yml")
endif()
# 如果有其他配置目录，也可以添加
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(
    DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
    FILES_MATCHING
    PATTERN "*.yml"
    PATTERN "*.yaml"
    PATTERN "*.xml"
    PATTERN "*.json")
endif()
# ===========================
# 导出配置
# ===========================
# 导出所有目标
install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake)

# 生成配置文件
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION share/${PROJECT_NAME}/cmake)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION share/${PROJECT_NAME}/cmake)

# ament 导出
ament_export_targets(${PROJECT_NAME}_Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV)

ament_package()
