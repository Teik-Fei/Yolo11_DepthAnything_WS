from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 1. Define Paths
    share = get_package_share_directory('yolo3d_stack')
    
    # Config Files
    yolo_params = os.path.join(share, 'config', 'params.yaml')
    nav2_params = os.path.join(share, 'config', 'nav2_params.yaml') 
    rviz_config = os.path.join(share, 'config', 'yolo3d.rviz')
    urdf_file = os.path.join(share, 'URDF', 'auv.urdf')

    # Read URDF
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        
        # ====================================================
        # 1. DRIVERS & ROBOT STATE
        # ====================================================
        
        # Robot State Publisher (Broadcasts URDF TF Tree)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
            output='screen'
        ),

        # Add this node to bridge Map -> Odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_odom_publisher',
            # Arguments: x y z yaw pitch roll parent_frame child_frame
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),

        # Camera Driver
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe', 
            name='usb_cam',
            parameters=[{
                'video_device': '/dev/video0',
                'framerate': 10.0,
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv',
                'frame_id': 'camera_link_optical' # <--- CRITICAL: Matches URDF
            }],
            remappings=[
                ('image_raw', '/camera/image_raw')
            ],
            output='screen'
        ),

        # Static TF: Map -> Base Link (Fakes the Odometry for testing)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['0', '0', '1.0', '0', '0', '0', 'map', 'base_link']
        ),

        # ====================================================
        # 2. PERCEPTION PIPELINE
        # ====================================================

        # YOLOv11 (TensorRT) - Generates Obstacle Cloud
        Node(
            package='yolo3d_stack',
            executable='yolo11_trt_node',
            name='yolo11_trt',
            output='screen',
            parameters=[{
            # YOLOv11 bounding box Engine Path
            #    "engine_path": "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/yolo11n_fp16.engine",
            # Yolov11-segmentation Engine Path
                "engine_path": "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/yolo11n-seg.engine",
                "image_topic": "/camera/image_raw",
                "visualize_output": True,
                "publish_debug_image": True,
                "debug_image_topic": "/yolo/debug_image",
                "conf_threshold": 0.75,
                "input_width": 640,
                "input_height": 640,
            }, yolo_params] 
        ),

        # Depth Anything (TensorRT)
        Node(
            package="yolo3d_stack",
            executable="depth_anything_trt_node",
            name="depth_anything_trt_node",
            output="screen",
            parameters=[yolo_params],
        ),

        # Fusion / BEV Visualization
        Node(
            package='yolo3d_stack',
            executable='fusion_bev_node', 
            name='fusion_bev_node',
            parameters=[yolo_params],
            output='screen'
        ), 

        # 3D Markers (For RViz)
        Node(
            package='yolo3d_stack',
            executable='yolo3d_markers_node',
            name='yolo3d_markers_node',
            parameters=[yolo_params],
            output='screen'
        ),

        # Depth -> Dense PointCloud (Visual V-Shape)
        #Node(
        #    package='yolo3d_stack',
        #    executable='depth_to_pointcloud_node', 
        #    name='depth_to_pointcloud_node',
        #    parameters=[yolo_params],
        #    output='screen'
        #),

        # ====================================================
        # 3. NAVIGATION STACK (NEW)
        # ====================================================
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='costmap',            # <--- MATCHES YAML
            output='screen',
            parameters=[nav2_params],
            remappings=[
                ('/costmap/get_state', '/costmap/get_state'),
                ('/costmap/change_state', '/costmap/change_state')
            ]
        ),

        # ====================================================
        # 4.LIFECYCLE MANAGER
        # ====================================================
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_costmap',
            output='screen',
            parameters=[
                {'use_sim_time': False},
                {'autostart': True},
                {'node_names': ['/costmap/costmap']}, # <--- MATCHES NODE NAME
                {'bond_timeout': 120.0}           # <--- INCREASED TIMEOUT (Fixes slow startup)
            ]
        ),

        # ====================================================
        # 5.VISUALIZATION
        # ====================================================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])