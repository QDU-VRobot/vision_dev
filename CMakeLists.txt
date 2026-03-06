cmake_minimum_required(VERSION 3.10)
project(vision_dart VERSION 1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(MVS REQUIRED)
find_package(Eigen3 REQUIRED)  # 添加 Eigen3 依赖

add_subdirectory(${CMAKE_SOURCE_DIR}/libxr)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(sdk ${SOURCES})

target_include_directories(
  sdk PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${MVS_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}  # 添加 Eigen3 头文件路径
)

target_link_libraries(
  sdk 
  xr
  ${OpenCV_LIBS}
  fmt::fmt
  spdlog::spdlog
  ${MVS_LIBS}
)

add_executable(serial_test test/serial_test.cpp src/serial_driver.cpp)

target_include_directories(serial_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(serial_test
    xr
    Eigen3::Eigen
)
