from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('yolo3d_stack')
    params = os.path.join(share, 'config', 'params.yaml')

    return LaunchDescription([
        # --- NEW: Camera Driver Node ---
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device': '/dev/video0',
                'framerate': 30.0,
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv' # Common for webcams
            }],
            remappings=[
                ('image_raw', '/camera/image_raw')
            ],
            output='screen'
        ),

        # Node 1: YOLO Object Detection
        Node(
            package='yolo3d_stack',
            executable='yolo11_node_exe',
            name='yolo11_node',
            parameters=[params],
            output='screen'
        ),
        
        # Node 2: Depth Estimation
        Node(
            package='yolo3d_stack',
            executable='depth_anything_node_exe',
            name='depth_anything_node',
            parameters=[params],
            output='screen'
        ),
        
        # Node 3: Fusion / BEV
        Node(
            package='yolo3d_stack',
            executable='fusion_bev_node_exe',
            name='fusion_bev_node',
            parameters=[params],
            output='screen'
        ),
    ])