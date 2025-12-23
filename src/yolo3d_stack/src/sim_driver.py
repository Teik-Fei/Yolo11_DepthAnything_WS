#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
import math
import time

# Correct Python TF library
from tf2_ros import TransformBroadcaster 

class FakeDriver(Node):
    def __init__(self):
        super().__init__('fake_driver')
        
        # Subscribe to Nav2 commands
        self.sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        
        # Publish Fake Sensor Data
        self.pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Update loop (20Hz)
        self.create_timer(0.05, self.update_pose) 
        
        # Robot State
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vth = 0.0
        self.last_time = self.get_clock().now()

    def cmd_callback(self, msg):
        # Update velocities from Nav2 command
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.vth = msg.angular.z

    def update_pose(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # 1. Calculate Movement (Kinematics)
        # Allows for strafing (y-axis) since it's an AUV
        delta_x = (self.vx * math.cos(self.th) - self.vy * math.sin(self.th)) * dt
        delta_y = (self.vx * math.sin(self.th) + self.vy * math.cos(self.th)) * dt
        delta_th = self.vth * dt

        self.x += delta_x
        self.y += delta_y
        self.th += delta_th

        # 2. Publish Odometry Message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = self.euler_to_quaternion(self.th)
        
        # Fake covariance so Nav2 trusts it
        odom.pose.covariance = [0.01 if i in [0, 7, 14, 21, 28, 35] else 0.0 for i in range(36)]
        
        self.pub.publish(odom)

        # 3. Publish TF (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = self.euler_to_quaternion(self.th)
        self.tf_broadcaster.sendTransform(t)

    def euler_to_quaternion(self, yaw):
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2)
        q.w = math.cos(yaw / 2)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = FakeDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()