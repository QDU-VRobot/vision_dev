cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX)

# add_definitions(-Wall -Werror)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

add_compile_options(-march=native -O3)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# ==========================================
# 1. 链接预编译好的 Rerun 静态库
# ==========================================
# 指向我们刚刚 install 出来的那个文件夹
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Visualisation/rerun_sdk_install")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../Visualisation/rerun_viewer/rerun" 
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")  # 将rerun渲染文程序复制到构建目录

# 现在 CMake 瞬间就能找到它，不需要再编译了！
find_package(rerun_sdk REQUIRED)


find_package(OpenCV REQUIRED )
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(MVS REQUIRED)
find_package(PkgConfig REQUIRED)
# 找到 OpenVINO 包
# 需要先 source openvino/setupvars.sh
find_package(OpenVINO REQUIRED)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(main 
    main.cpp
    src/HikCamera.cpp
    src/logger.cpp
    src/Detecor.cpp
    # src/NumClassifier.cpp
    src/Solver.cpp
    src/Target.cpp
    src/Tracker.cpp
    src/EKFKalman.cpp
    ./Function.cpp
)

add_executable(smallmain 
    smallmain.cpp
    src/HikCamera.cpp
    src/logger.cpp
    src/Detecor.cpp
    src/SmallNumClassifier.cpp
    src/Solver.cpp
    src/Target.cpp
    src/Tracker.cpp
    src/EKFKalman.cpp
    ./Function.cpp
)

# add_executable(Yolomain 
#     Yolomain.cpp
#     src/HikCamera.cpp
#     src/logger.cpp
#     src/Solver.cpp
#     src/Target.cpp
#     src/EKFKalman.cpp
#     ./Function.cpp
# )
target_include_directories(main PUBLIC ${MVS_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} openvino::runtime rerun_sdk)

target_include_directories(smallmain PUBLIC ${MVS_INCLUDE_DIRS})
target_link_libraries(smallmain ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} rerun_sdk)

#target_include_directories(Yolomain PUBLIC ${MVS_INCLUDE_DIRS})
#target_link_libraries(Yolomain ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} openvino::runtime rerun_sdk)






