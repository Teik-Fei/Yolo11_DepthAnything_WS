cmake_minimum_required(VERSION 3.16)
project(yolo3d_stack)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)


# OpenCV with CUDA build on Jetson
find_package(OpenCV REQUIRED)


include_directories(
include
${OpenCV_INCLUDE_DIRS}
)


add_executable(yolo11_node src/yolo11_node.cpp)
ament_target_dependencies(yolo11_node rclcpp sensor_msgs std_msgs image_transport cv_bridge vision_msgs)
target_link_libraries(yolo11_node ${OpenCV_LIBS})


add_executable(depth_anything_node src/depth_anything_node.cpp)
ament_target_dependencies(depth_anything_node rclcpp sensor_msgs std_msgs image_transport cv_bridge)
target_link_libraries(depth_anything_node ${OpenCV_LIBS})


add_executable(fusion_bev_node src/fusion_bev_node.cpp)
ament_target_dependencies(fusion_bev_node rclcpp sensor_msgs std_msgs image_transport cv_bridge vision_msgs message_filters)
target_link_libraries(fusion_bev_node ${OpenCV_LIBS})


install(TARGETS
yolo11_node
depth_anything_node
fusion_bev_node
DESTINATION lib/${PROJECT_NAME})


install(DIRECTORY launch config
DESTINATION share/${PROJECT_NAME})


ament_package()
