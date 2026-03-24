cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV REQUIRED )
# 找到 OpenVINO 包
# 需要先 source openvino/setupvars.sh


add_compile_options(-g -O3)
# find_package(LibUSB 1.0 REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(test 
    test.cpp
    ../../rm-main/src/Target.cpp
)
target_link_libraries(test ${OpenCV_LIBS})






