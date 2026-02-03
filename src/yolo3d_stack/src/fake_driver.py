#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import sys, select, termios, tty
import math
import threading

# Keys for movement
msg = """
Control Your AUV!
---------------------------
Moving around:
   w
a  s  d
   x

w/x : move forward/backward (X axis)
a/d : rotate yaw
s   : stop

CTRL-C to quit
"""

class FakeDriver(Node):
    def __init__(self):
        super().__init__('fake_driver')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.publish_tf) # 20Hz
        
        # Robot State
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        
        self.get_logger().info("Fake Driver Started. Use WASD to move.")

    def publish_tf(self):
        # Update Position
        self.th += self.turn * 0.05
        self.x += self.speed * math.cos(self.th) * 0.05
        self.y += self.speed * math.sin(self.th) * 0.05

        # Create TF Message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0 # Swim level

        # Quaternion from Yaw
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(self.th / 2.0)
        t.transform.rotation.w = math.cos(self.th / 2.0)

        self.br.sendTransform(t)

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    rclpy.init()
    settings = termios.tcgetattr(sys.stdin)
    node = FakeDriver()
    
    # Spin in a separate thread so input doesn't block TF
    spinner = threading.Thread(target=rclpy.spin, args=(node,))
    spinner.start()

    try:
        print(msg)
        while True:
            key = getKey(settings)
            if key == 'w':
                node.speed = 0.5
            elif key == 'x':
                node.speed = -0.5
            elif key == 'a':
                node.turn = 1.0
            elif key == 'd':
                node.turn = -1.0
            elif key == 's':
                node.speed = 0.0
                node.turn = 0.0
            elif key == '\x03': # Ctrl+C
                break
    except Exception as e:
        print(e)
    finally:
        node.speed = 0
        node.turn = 0
        rclpy.shutdown()
        spinner.join()

if __name__ == '__main__':
    main()