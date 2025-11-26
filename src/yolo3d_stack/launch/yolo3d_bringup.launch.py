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
                'framerate': 5.0,
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv' 
            }],
            remappings=[
                ('image_raw', '/camera/image_raw')
            ],
            output='screen'
        ),

        # --- Node 1: YOLO Object Detection ---
        Node(
            package='yolo3d_stack',
            executable='yolo11_node',  # Removed '_exe' to match CMake
            name='yolo11_node',
            parameters=[params],
            output='screen'
        ),
        
        # --- Node 2: Depth Estimation ---
        Node(
            package='yolo3d_stack',
            executable='depth_anything_node', # Removed '_exe'
            name='depth_anything_node',
            parameters=[params, {'depth_factor': 10.0}],
            output='screen'
        ),
        
        # --- Node 3: Fusion / BEV ---
        #Node(
        #    package='yolo3d_stack',
        #    executable='fusion_bev_node', # Removed '_exe'
        #    name='fusion_bev_node',
        #    parameters=[params],
        #    output='screen'
        #), 

        # --- Node 4: 3D Markers ---
        Node(
            package='yolo3d_stack',
            executable='yolo3d_markers_node', # Removed '_exe'
            name='yolo3d_markers_node',
            parameters=[params],
            output='screen'
        ),

        # --- Node 5: Depth → PointCloud ---
        Node(
            package='yolo3d_stack',
            executable='depth_to_pointcloud_node', # Removed '_exe'
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
    ])