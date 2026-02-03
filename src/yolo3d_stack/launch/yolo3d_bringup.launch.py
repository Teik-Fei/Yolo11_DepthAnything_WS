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


       # Bridge Map -> Odom (Static for now)
       Node(
           package='tf2_ros',
           executable='static_transform_publisher',
           name='static_map_odom_publisher',
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
               'frame_id': 'camera_link_optical',
               'camera_info_url': 'package://yolo3d_stack/config/calibration.yaml'
           }],
           remappings=[
               ('image_raw', '/camera/image_raw'),
               # ENSURE THIS REMAPPING EXISTS SO RTABMAP FINDS IT:
               ('camera_info', '/camera/camera_info')
           ],
           output='screen'
       ),


       # 1. YUYV -> BGR Converter (Fixes Odometry)
       Node(
           package='yolo3d_stack',
           executable='image_converter_node', # Make sure this matches your build name!
           name='image_converter',
           output='screen'
       ),


       # 2. PointCloud Cleaner (Fixes OctoMap Crash)
       Node(
           package='yolo3d_stack',
           executable='cloud_cleaner_node',   # Make sure this matches your build name!
           name='cloud_cleaner',
           output='screen'
       ),


       # Static TF: odom -> Base Link (Fake Odometry) [commented out for simulation purpose]
       Node(
           package='tf2_ros',
           executable='static_transform_publisher',
           arguments = ['0', '0', '1.0', '0', '0', '0', 'odom', 'base_link']
       ),


       # ====================================================
       # 2. PERCEPTION PIPELINE
       # ====================================================


       # YOLOv11 (TensorRT)
       #Node(
       #    package='yolo3d_stack',
       #    executable='yolo11_trt_node',
       #    name='yolo11_trt',
       #    output='screen',
       #    parameters=[{
       #        "engine_path": "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/yolo11n-seg.engine",
       #        "image_topic": "/camera/image_raw",
       #        "visualize_output": True,
       #        "publish_debug_image": True,
       #        "debug_image_topic": "/yolo/debug_image",
       #        "conf_threshold": 0.75,
       #        "input_width": 640,
       #        "input_height": 640,
       #    }, yolo_params]
       #),


       # SAUVC dataset (TensorRT)
       #Node(
       #    package='yolo3d_stack',
       #    executable='yolo11_trt_node',
       #    name='yolo11_trt',
       #    output='screen',
       #    parameters=[{
       #        "engine_path": "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/sauvc_5.engine",
       #        "image_topic": "/camera/image_raw",
       #        "visualize_output": True,
       #        "publish_debug_image": True,
       #        "debug_image_topic": "/yolo/debug_image",
       #        "conf_threshold": 0.75,
       #        "input_width": 640,
       #        "input_height": 640,
       #    }, yolo_params]
       #),

       # SAUVC dataset (TensorRT) - UPDATED
        Node(
            package='yolo3d_stack',
            executable='yolo11_trt_node',
            name='yolo11_trt',
            output='screen',
            remappings=[
                ('/camera/image_raw', '/camera/image_raw'),      # Input RGB
                ('/depth/image_raw', '/depth/image_raw'),        # Input Depth
                ('/yolo/obstacle_cloud', '/yolo/obstacle_cloud') # Output Cloud
            ],
            parameters=[{
                "engine_path": "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/sauvc_5.engine",
                "conf_threshold": 0.45,
                "visualize_output": True
            }]
        ),


       # Depth Anything (TensorRT) - THE BRAIN
       Node(
           package="yolo3d_stack",
           executable="depth_anything_trt_node",
           name="depth_anything_trt_node",
           output="screen",
           parameters=[
               yolo_params,
               # OVERRIDE: Adjust this value to calibrate the "LiDAR" range!
               # If the wall is 2m away but Rviz shows 1m, double this to ~2.5
               {"max_depth_m": 1.5}
           ],
       ),


       # Depth -> Dense PointCloud - THE GEOMETRY
       Node(
          package='yolo3d_stack',
          executable='depth_to_pointcloud_node',
          name='depth_to_pointcloud_node',
          parameters=[
              yolo_params,
              # Downsample to reduce CPU load (LiDARs don't need HD resolution)
              {"pixel_step": 4},
              {"fx": 650.0},
              {"fy": 650.0}
          ],
          output='screen'
       ),

       # ====================================================
       # PointCloud to LaserScan (Slower & Noisy)
       # PointCloud -> LaserScan - THE SLICER (Virtual LiDAR)
       # ====================================================
       #Node(
       #    package='pointcloud_to_laserscan',
       #    executable='pointcloud_to_laserscan_node',
       #    name='pointcloud_to_laserscan',
       #    remappings=[
       #        ('cloud_in', '/depth/pointcloud'),
       #        ('scan', '/scan')
       #    ],
       #    parameters=[{
       #        'target_frame': 'base_link', # Projects scan onto the robot base
       #        'transform_tolerance': 0.01,
       #        'min_height': -3.0, # Floor cut-off (relative to camera height) [might need to fine tune this based on diff settings]
       #        'max_height': 3.0,  # Ceiling cut-off [might need to fine tune this based on diff settings]
       #        'angle_min': -1.57, # -90 deg
       #        'angle_max': 1.57,  # +90 deg
       #        'range_min': 0.5,   # Ignore noise right in front of lens
       #        'range_max': 5.0,   # Max reliable range
       #        'use_inf': True
       #    }]
       #),


       # ====================================================
       # VIRTUAL LASERSCAN (Fast & Clean)
       # create a line for detection instead of full pointcloud conversion
       # ====================================================
      
       # 1. The TF Fix: Rotates the scan 90deg so it lays flat on the "floor"
       #    Connects 'camera_link_optical' -> 'virtual_laser_frame'
       #    Args: x y z yaw pitch roll
       Node(
           package='tf2_ros',
           executable='static_transform_publisher',
           name='virtual_laser_tf',
           arguments=['0', '0', '0', '0', '0', '1.57', 'camera_link_optical', 'virtual_laser_frame'],
           output='screen'
       ),


       # 2. The Node: Slices the Depth Image into a LaserScan
       Node(
           package='yolo3d_stack',
           executable='virtual_laserscan_node',
           name='virtual_laserscan_node',
           output='screen',
           parameters=[{
               'depth_topic': '/depth/image_raw',
               'scan_topic':  '/scan',
               'scan_height': 10,       # Slice thickness (pixels)
               'scan_line_idx': -1,     # -1 = Center of image
               'min_range': 0.2,
               'max_range': 5.0,       # Max valid range (meters)
              
               # IMPORTANT: Match these to your Depth/Camera params!
               # (Using the values seen in your depth_to_pointcloud config)
               'fx': 650.0,
               'cx': 320.0
           }]
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


       # ====================================================
       # 3. NAVIGATION STACK
       # ====================================================
       Node(
           package='nav2_costmap_2d',
           executable='nav2_costmap_2d',
           name='costmap',
           namespace='costmap',
           output='screen',
           parameters=[nav2_params],
           remappings=[
               ('/costmap/get_state', '/costmap/get_state'),
               ('/costmap/change_state', '/costmap/change_state')
           ]
       ),


       # ====================================================
       # VISUAL ODOMETRY (The "GPS" for your map) [commented out for simulation purposes]
       # ====================================================
       #Node(
       #    package='rtabmap_odom',
       #    executable='rgbd_odometry',
       #    name='visual_odometry',
       #    output='screen',
       #    parameters=[{
       #        'frame_id': 'base_link',
       #        'odom_frame_id': 'odom',
       #        'publish_tf': True,
       #        'approx_sync': True,
       #        'approx_sync_max_interval': 0.3, # 0.3s is perfect for AI lag
       #        'wait_for_transform': 0.3,
       #        'queue_size': 50,


               # --- Tuning Fixes ---
       #        'Odom/Strategy': '1',          # CHANGED: Frame-to-Frame (Much more stable startup)
       #        'Vis/MinInliers': '8',         # CHANGED: Lower threshold to accept more frames
       #        'Vis/FeatureType': '2',        # CHANGED: ORB (Better general performance than GFTT)
       #        'Vis/MaxDepth': '5.0',         # Increased slightly to see further
       #        'Reg/Force3DoF': 'False',
       #        'Odom/ResetCountdown': '1',    # NEW: If lost, try to reset immediately
       #    }],
       #    remappings=[
       #        ('rgb/image', '/camera/image_bgr'),         # Input RGB
       #        ('depth/image', '/depth/image_raw'),        # Input AI Depth
       #        ('rgb/camera_info', '/camera/camera_info'), # Input Calibration
       #        ('odom', '/odom')                           # Output Topic
       #    ]
       #),


       # ====================================================
       # 3D MAPPING (OCTOMAP)
       # ====================================================
       Node(
           package='octomap_server',
           executable='octomap_server_node',
           name='octomap_server',
           output='screen',
           parameters=[{
               'resolution': 0.1,                 # Voxel size (10cm cubes)
               'frame_id': 'odom',                 # Global frame
               'base_frame_id': 'base_link',      # Robot frame
               'sensor_model/max_range': 5.0,     # Ignore depth noise > 5m
              
               # FILTERING (Critical for Underwater)
               'sensor_model/hit': 0.7,           # Confidence increase if obstacle seen
               'sensor_model/miss': 0.2,          # Confidence decrease if empty space seen
               'sensor_model/min': 0.12,          # Clamp min probability
               'sensor_model/max': 0.97,          # Clamp max probability
              
               'filter_ground': False,            # Set True if you want to remove the sea floor
               # --- FIX TIMING ISSUES ---
               'transform_tolerance': 1.0,      # Allow 1.0s lag
           }],
           remappings=[
               ('cloud_in', '/yolo/obstacle_cloud')
           ]
       ),


       # ====================================================
       # 4. LIFECYCLE MANAGER
       # ====================================================
       Node(
           package='nav2_lifecycle_manager',
           executable='lifecycle_manager',
           name='lifecycle_manager_costmap',
           output='screen',
           parameters=[
               {'use_sim_time': False},
               {'autostart': True},
               {'node_names': ['/costmap/costmap']},
               {'bond_timeout': 120.0}
           ]
       ),


       # ====================================================
       # 5. VISUALIZATION
       # ====================================================
       Node(
           package='rviz2',
           executable='rviz2',
           name='rviz2',
           arguments=['-d', rviz_config],
           output='screen'
       ),
   ])