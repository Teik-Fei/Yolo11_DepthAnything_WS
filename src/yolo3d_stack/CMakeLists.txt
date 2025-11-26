cmake_minimum_required(VERSION 3.8)
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

ament_package()