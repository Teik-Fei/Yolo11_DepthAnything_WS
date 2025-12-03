cmake_minimum_required(VERSION 3.10)
project(yolo3d_stack)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# ======================
# CPU vs CUDA toggle
# ======================
option(USE_CUDA "Build with CUDA + TensorRT" OFF)

# ======================
# Common ROS2 packages
# ======================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)

include_directories(include)

# ===========================================
# 1. CPU VERSION
# ===========================================
if(NOT USE_CUDA)
    message(STATUS "Building CPU version")

    add_executable(yolo11_node src/yolo11_node.cpp)
    add_executable(depth_anything_node src/depth_anything_node.cpp)
    add_executable(fusion_bev_node src/fusion_bev_node.cpp)
    add_executable(depth_to_pointcloud_node src/depth_to_pointcloud_node.cpp)
    add_executable(yolo3d_markers_node src/yolo3d_markers_node.cpp)

    foreach(target
        yolo11_node
        depth_anything_node
        fusion_bev_node
        depth_to_pointcloud_node
        yolo3d_markers_node)

        ament_target_dependencies(${target}
            rclcpp rclcpp_components
            sensor_msgs cv_bridge
            vision_msgs visualization_msgs
            OpenCV
        )
    endforeach()

    set(BUILT_TARGETS
        yolo11_node
        depth_anything_node
        fusion_bev_node
        depth_to_pointcloud_node
        yolo3d_markers_node
    )
endif()

# ===========================================
# 2. CUDA VERSION (Optional)
# ===========================================
if(USE_CUDA)
    message(STATUS "Building CUDA + TensorRT version")

    find_path(CUDA_INCLUDE_DIR cuda_runtime.h PATHS /usr/local/cuda/include /usr/include)
    find_library(CUDART_LIB cudart PATHS /usr/local/cuda/lib64 /usr/lib)

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/local/include)
    find_library(NVINFER_LIB nvinfer PATHS /usr/lib /usr/local/lib)
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS /usr/lib /usr/local/lib)

    if(NOT CUDART_LIB OR NOT NVINFER_LIB)
        message(FATAL_ERROR "CUDA or TensorRT not found.")
    endif()

    include_directories(${CUDA_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})

    add_executable(yolo11_trt_node src/yolo11_trt_node.cpp)
    add_executable(depth_anything_node src/depth_anything_node.cpp)
    add_executable(fusion_bev_node src/fusion_bev_node.cpp)
    add_executable(depth_to_pointcloud_node src/depth_to_pointcloud_node.cpp)
    add_executable(yolo3d_markers_node src/yolo3d_markers_node.cpp)

    foreach(target
        yolo11_trt_node
        depth_anything_node
        fusion_bev_node
        depth_to_pointcloud_node
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

    set(BUILT_TARGETS
        yolo11_trt_node
        depth_anything_node
        fusion_bev_node
        depth_to_pointcloud_node
        yolo3d_markers_node
    )
endif()

# ===========================================
# 3. Install only the built targets
# ===========================================
install(TARGETS
    ${BUILT_TARGETS}
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ launch config models
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
