from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Ensure this path exists in your install folder!
    share = get_package_share_directory('yolo3d_stack')
    params = os.path.join(share, 'config', 'params.yaml')

    return LaunchDescription([
        # --- Camera Driver Node ---
        # Note: Ensure 'sudo apt install ros-humble-usb-cam' is run
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe', 
            name='usb_cam',
            parameters=[{
                'video_device': '/dev/video0',
                'framerate': 10.0,
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv' 
            }],
            remappings=[
                ('image_raw', '/camera/image_raw')
            ],
            output='screen'
        ),

        # --- Node 1: YOLO Object Detection using CPU---
        #Node(
        #    package='yolo3d_stack',
        #    executable='yolo11_node', 
        #    name='yolo11_node',
        #    parameters=[params],
        #    output='screen'
        #),

        # Modified Node 1 (using tensorRT)
        Node(
            package='yolo3d_stack',
            executable='yolo11_trt_node',   # TensorRT CUDA node
            name='yolo11_trt',
            output='screen',
            parameters=[{
                "engine_path": "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/yolo11n_fp16.engine",
                "image_topic": "/camera/image_raw",

                # ADD THESE
                "visualize_output": True,
                "publish_debug_image": True,
                "debug_image_topic": "/yolo/debug_image",

                "conf_threshold": 0.75,
                "input_width": 640,
                "input_height": 640
            }]
        ),

        # --- Node 2: Depth Estimation ---
        #Node(
        #    package='yolo3d_stack',
        #    executable='depth_anything_node', 
        #    name='depth_anything_node',
        #    parameters=[params, {'depth_factor': 0.75}],
        #    output='screen'
        #),
        
        # --- Node 3: Fusion / BEV ---
        Node(
            package='yolo3d_stack',
            executable='fusion_bev_node', 
            name='fusion_bev_node',
            parameters=[params],
            output='screen'
        ), 

        # --- Node 4: 3D Markers ---
        Node(
            package='yolo3d_stack',
            executable='yolo3d_markers_node',
            name='yolo3d_markers_node',
            parameters=[params],
            output='screen'
        ),

        # --- Node 5: Depth → PointCloud ---
        Node(
            package='yolo3d_stack',
            executable='depth_to_pointcloud_node', 
            name='depth_to_pointcloud_node',
            parameters=[params],
            output='screen'
        ),

        # --- NEW: Rviz2 with Config ---
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join('/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/config', 'yolo3d.rviz')],
            output='screen'
        ),

        # --- Static TF Publisher (World -> Camera) ---
        # This tells RViz where the camera is in the 3D world
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['0', '0', '1.0', '0', '0', '0', 'map', 'default_cam'], 
            # Note: Replace 'camera_frame_id' with the actual frame_id 
            # your camera publishes (usually 'camera_link' or 'camera_color_optical_frame')
        ),

        # --- Depth Anything TensorRT Node ---
        Node(
            package="yolo3d_stack",
            executable="depth_anything_trt_node",
            name="depth_anything_trt_node",
            output="screen",
            parameters=["config/depth_anything_trt.yaml"],
        )
    ])