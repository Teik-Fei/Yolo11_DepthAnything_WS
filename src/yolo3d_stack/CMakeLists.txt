## Cmake file for running yolo3d_stack on CPU

#[[cmake_minimum_required(VERSION 3.8)
project(yolo3d_stack)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================================
# 1. Find Dependencies
# =========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

# =========================================
# 2. Build Executables
# =========================================

# --- Node A: YOLO11 Node (Object Detection) ---
add_executable(yolo11_node src/yolo11_node.cpp)
target_include_directories(yolo11_node PUBLIC include)
ament_target_dependencies(yolo11_node
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  cv_bridge
  OpenCV
)

# --- Node B: Depth Anything Node (Depth Estimation) ---
add_executable(depth_anything_node src/depth_anything_node.cpp)
target_include_directories(depth_anything_node PUBLIC include)
ament_target_dependencies(depth_anything_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  OpenCV
)

# --- Node C: Fusion BEV Node (Bird's Eye View) ---
add_executable(fusion_bev_node src/fusion_bev_node.cpp)
target_include_directories(fusion_bev_node PUBLIC include)
ament_target_dependencies(fusion_bev_node
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  cv_bridge
  OpenCV
)

# --- Node D: Depth to PointCloud (Projection) ---
add_executable(depth_to_pointcloud_node src/depth_to_pointcloud_node.cpp)
target_include_directories(depth_to_pointcloud_node PUBLIC include)
ament_target_dependencies(depth_to_pointcloud_node
  rclcpp
  sensor_msgs
  cv_bridge
)

# --- Node E: YOLO3D Markers (Visualization) ---
add_executable(yolo3d_markers_node src/yolo3d_markers_node.cpp)
target_include_directories(yolo3d_markers_node PUBLIC include)
ament_target_dependencies(yolo3d_markers_node
  rclcpp
  vision_msgs
  visualization_msgs
)

# =========================================
# 3. Install Rules
# =========================================
install(TARGETS
  yolo11_node
  depth_anything_node
  fusion_bev_node
  depth_to_pointcloud_node
  yolo3d_markers_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install launch directory
# FIX: Moved OPTIONAL before FILES_MATCHING
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
  FILES_MATCHING PATTERN "*.py"
)

ament_package()]]


## Cmake file for running yolo3d_stack with CUDA + TensorRT
cmake_minimum_required(VERSION 3.10)
project(yolo3d_stack)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# FIND ROS2 PACKAGES
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(message_filters REQUIRED)

# CUDA + TensorRT (Automatic Search)
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    PATHS /usr/local/cuda/include /usr/include /usr/include/aarch64-linux-gnu
)

find_library(CUDART_LIB libcudart.so
    PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include /usr/include/aarch64-linux-gnu /usr/local/include
)

find_library(NVINFER_LIB nvinfer
    PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib
)

find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
    PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib
)

# STOP BUILD IF CUDA/TRT NOT FOUND
if (NOT CUDART_LIB)
    message(FATAL_ERROR "❌ CUDA Runtime (cudart) not found.\nInstall:\n sudo apt install cuda-toolkit")
endif()

if (NOT NVINFER_LIB)
    message(FATAL_ERROR "❌ TensorRT not found.\nInstall:\n sudo apt install tensorrt-dev")
endif()

include_directories(
    include
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# BUILD ALL EXECUTABLES
add_executable(yolo11_trt_node              src/yolo11_trt_node.cpp)
add_executable(depth_anything_node          src/depth_anything_node.cpp)
add_executable(depth_to_pointcloud_node     src/depth_to_pointcloud_node.cpp)
add_executable(fusion_bev_node              src/fusion_bev_node.cpp)
add_executable(yolo3d_markers_node          src/yolo3d_markers_node.cpp)

# LINK LIBS
foreach(target 
    yolo11_trt_node 
    depth_anything_node 
    depth_to_pointcloud_node 
    fusion_bev_node 
    yolo3d_markers_node)

    ament_target_dependencies(${target}
        rclcpp sensor_msgs std_msgs
        cv_bridge vision_msgs visualization_msgs
        image_transport message_filters
    )

    target_link_libraries(${target}
        ${OpenCV_LIBS}
        ${CUDART_LIB}
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
    )
endforeach()

# INSTALL
install(TARGETS
    yolo11_trt_node
    depth_anything_node
    depth_to_pointcloud_node
    fusion_bev_node
    yolo3d_markers_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config models
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
