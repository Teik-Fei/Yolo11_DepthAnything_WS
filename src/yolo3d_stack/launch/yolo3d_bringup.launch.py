from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
share = get_package_share_directory('yolo3d_stack')
params = os.path.join(share, 'config', 'params.yaml')


return LaunchDescription([
Node(
package='yolo3d_stack',
executable='yolo11_node',
name='yolo11_node',
parameters=[params],
output='screen'
),
Node(
package='yolo3d_stack',
executable='depth_anything_node',
name='depth_anything_node',
parameters=[params],
output='screen'
),
Node(
package='yolo3d_stack',
executable='fusion_bev_node',
name='fusion_bev_node',
parameters=[params],
output='screen'
),
])
