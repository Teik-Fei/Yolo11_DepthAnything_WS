cmake_minimum_required(VERSION 3.8)
project(yolo3d_stack)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find Dependencies
find_package(message_filters REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# 2. Include Directories
include_directories(include)

# === Jetson-specific vision_msgs include path fix ===
# The official NVIDIA ROS build does NOT export include directories
include_directories(/opt/ros/humble/include)
include_directories(/opt/ros/humble/include/vision_msgs)
include_directories(/opt/ros/humble/include/vision_msgs/vision_msgs)


# --- HELPER MACRO ---
macro(configure_ros_node node_name source_file class_name)
  add_library(${node_name} SHARED ${source_file})
  
  target_include_directories(${node_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${vision_msgs_INCLUDE_DIRS}
    ${image_transport_INCLUDE_DIRS} 
  )
  
  ament_target_dependencies(${node_name}
    rclcpp
    rclcpp_components
    sensor_msgs
    cv_bridge
    vision_msgs
    image_transport
    OpenCV
    message_filters
  )
  
  rclcpp_components_register_node(${node_name} PLUGIN "${class_name}" EXECUTABLE ${node_name}_exe)
endmacro()

# 3. Configure Nodes
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/depth_anything_node.cpp")
    configure_ros_node(depth_anything_node src/depth_anything_node.cpp "DepthAnythingNode")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_node.cpp")
    configure_ros_node(yolo11_node src/yolo11_node.cpp "Yolo11Node")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/fusion_bev_node.cpp")
    configure_ros_node(fusion_bev_node src/fusion_bev_node.cpp "FusionBevNode")
endif()

# 4. Install Executables/Libraries
install(TARGETS 
  depth_anything_node
  yolo11_node
  fusion_bev_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 5. Install Headers
install(DIRECTORY include/
  DESTINATION include
)

# 6. CRITICAL FIX: Install Launch and Config folders
# This allows "ros2 launch" to find your python file and params.yaml
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()