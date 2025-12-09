# 🌊 YOLO11-Seg + Depth Anything + Octomap Fusion (ROS 2 Humble)

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin-green)
![Acceleration](https://img.shields.io/badge/Acceleration-TensorRT%20%7C%20CUDA-orange)

A complete **3D Perception, Mapping, and Navigation Stack** for Autonomous Underwater Vehicles (AUVs).

This package upgrades standard 2D detection into a full **volumetric navigation system**. It fuses **YOLO11-Seg (Instance Segmentation)** with **Depth Anything V2** to create precise 3D object clouds, builds a persistent voxel map using **Octomap**, and feeds it directly into **Nav2** for obstacle avoidance.

---

## ✨ Key Features

### 👁️ Perception
* **Instance Segmentation:** Uses YOLO11-Seg to create precise masks (not just boxes) of underwater objects.
* **Masked Depth Cloud:** Projects *only* the object's pixels into 3D space, ignoring background noise.
* **Underwater Preprocessing:** Includes real-time **CLAHE** & **White Balance** to restore color and contrast in murky water.

### 🧠 Mapping & Memory
* **Volumetric Mapping (Octomap):** Builds a persistent 3D voxel grid. The robot "remembers" walls and gates even when it turns away.
* **Visual Odometry:** Integrated **RTAB-Map** for position tracking without external GPS/DVL.

### 🧭 Navigation
* **Nav2 Integration:** Custom costmap layers that respect both:
    * **Reflex:** Instant "Safety Bubbles" from YOLO detections.
    * **Memory:** Persistent obstacles from Octomap.

---

## 🛠️ System Requirements
* **Hardware:** NVIDIA Jetson Orin NX.
* **OS:** Ubuntu 22.04 (ROS 2 Humble).
* **Camera:** USB Camera (Calibrated for underwater usage).
* **Power:** Requires **15W Mode** (`sudo nvpmodel -m 1`) or a high-current battery (5A+) to prevent throttling.

---

## 📦 Installation

1.  **Clone & Install Dependencies**
    ```bash
    mkdir -p ~/yolo3d_ws/src
    cd ~/yolo3d_ws/src
    git clone <https://github.com/Teik-Fei/Yolo11_DepthAnything_WS.git>
    
    # Install ROS packages
    sudo apt install ros-humble-octomap-server ros-humble-rtabmap-odom ros-humble-nav2-bringup ros-humble-pcl-conversions
    ```

2.  **Model Preparation**
    You need **two** TensorRT engines:
    * **YOLO11-Seg:** Export your segmentation model:
        ```bash
        yolo export model=yolo11n-seg.pt format=engine half=True device=0
        ```
    * **Depth Anything V2:** Convert the ONNX model using `trtexec`:
        ```bash
        /usr/src/tensorrt/bin/trtexec --onnx=depth_anything_v2_vits.onnx --saveEngine=depth_anything_v2_vits_fp16.engine --fp16
        ```

3.  **Build**
    ```bash
    cd ~/yolo3d_ws
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    source install/setup.bash
    ```

---

## 🚀 Usage

### 1. The "Master Switch" (Pool Mode)
To run the full stack (Drivers + Perception + Mapping + Odometry):
```bash
ros2 launch yolo3d_stack yolo3d_bringup.launch.py
```
---

## 2. Modes of Operation
You can edit `yolo3d_bringup.launch.py` to switch modes:

* **Bench Test (Dry):**

    * Enable static_transform_publisher (Lines 70-75).

    * Disable rtabmap_odom.

    * Result: Robot stays at (0,0,0) but mapping works for testing.

* **Pool Test (Wet):**

* Disable static_transform_publisher.

* Enable rtabmap_odom (Lines 180+).

* Result: Robot tracks its own movement using floor texture.
---

## **🖥️ Visualization Guide (RViz)**
The system outputs multiple layers of reality:

1. 🟣 Purple Spheres: Instant YOLO detections (Safety Bubbles).

2. 🟦 Blue/Colored Blocks: Octomap Voxels (Long-term memory).

3. ⬜ White Dots: Masked 3D PointCloud (Precise object shape).

4. 🟩 Green Box: Real-time 2D detection overlay (/yolo/debug_image).
---

## 🎮 Simulation Mode (Fake Driver)

This stack includes a **"Fake Driver"** utility that allows you to fly the AUV in RViz using your keyboard (WASD). This is useful for testing the OctoMap integration and TF tree without physical movement.

### 1. Disable Real Odometry
Before running the simulation, you must temporarily disable the real Visual Odometry node to prevent TF conflicts.
* Open `src/yolo3d_stack/launch/yolo3d_bringup.launch.py`
* Comment out the `rtabmap_odom` node block.

### 2. Create the Driver Script
Create the driver file at `src/yolo3d_stack/yolo3d_stack/fake_driver.py`:

<details>
<summary>Click to see fake_driver.py code</summary>

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import sys, select, termios, tty
import math
import threading

msg = """
Control Your AUV!
---------------------------
w/x : move forward/backward
a/d : rotate yaw
s   : stop
CTRL-C to quit
"""

class FakeDriver(Node):
    def __init__(self):
        super().__init__('fake_driver')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.publish_tf) 
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0

    def publish_tf(self):
        self.th += self.turn * 0.05
        self.x += self.speed * math.cos(self.th) * 0.05
        self.y += self.speed * math.sin(self.th) * 0.05

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = math.sin(self.th / 2.0)
        t.transform.rotation.w = math.cos(self.th / 2.0)
        self.br.sendTransform(t)

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    return sys.stdin.read(1) if rlist else ''

def main():
    rclpy.init()
    settings = termios.tcgetattr(sys.stdin)
    node = FakeDriver()
    spinner = threading.Thread(target=rclpy.spin, args=(node,))
    spinner.start()
    try:
        print(msg)
        while True:
            key = getKey(settings)
            if key == 'w': node.speed = 0.5
            elif key == 'x': node.speed = -0.5
            elif key == 'a': node.turn = 1.0
            elif key == 'd': node.turn = -1.0
            elif key == 's': node.speed = 0.0; node.turn = 0.0
            elif key == '\x03': break
    finally:
        rclpy.shutdown()
        spinner.join()

if __name__ == '__main__':
    main()

```
</details>


## 3. Usage
1. **Launch the Stack:**

```Bash
ros2 launch yolo3d_stack yolo3d_bringup.launch.py
```
2. **Run the Fake Driver (in a new terminal):**

```Bash
python3 src/yolo3d_stack/yolo3d_stack/fake_driver.py
```
3. Controls:

* Click on the terminal running the python script.

* W / X: Move Forward / Backward

* A / D: Rotate Left / Right

* S: Stop

As you "move" the robot, the static camera view will be painted into the 3D map (creating a tunnel effect), confirming that mapping is working.

## **⚙️ Configuration & Tuning**
1. Underwater Camera Calibration (`params.yaml`)  
Light refracts differently in water. Standard air calibration **will fail.**

```yaml
yolo11_trt:
  ros__parameters:
    # Approximate 1.33x zoom effect of water
    fx: 740.5 
    fy: 740.5
    # Lower confidence for murky water
    confidence_threshold: 0.50
```

## **Navigation Safety (`nav2_params.yaml`)**
* Inflation Radius: Set to 0.5 to keep the robot 50cm away from any blue block.

* Observation Sources: Ensure octomap_voxels is listed so the robot respects the memory map.

---

## 🧩 Nodes Description

| Node Name | Role | Function |
| :--- | :--- | :--- |
| **`yolo11_trt_node`** | **Perception Core** | Runs YOLO11-Seg (TensorRT). Performs underwater preprocessing (CLAHE), extracts 3D coordinates, and publishes the **Masked PointCloud** for mapping. |
| **`depth_anything_trt_node`** | **Depth Engine** | Runs Depth Anything V2 (TensorRT) to generate a high-quality depth map from the monocular RGB camera. |
| **`octomap_server`** | **Mapping** | Consumes the masked point cloud to build the persistent **3D Voxel Map** (Blue Blocks) used for navigation memory. |
| **`rtabmap_odom`** | **Localization** | Performs **Visual Odometry**. Tracks the robot's movement relative to the pool floor to publish the `odom` → `base_link` transform. |
| **`nav2_costmap`** | **Navigation** | Manages the **Global** and **Local** costmaps. It marks the blue Octomap blocks as "Lethal Obstacles" so the robot avoids them. |
| **`fusion_bev_node`** | **Visualization** | Generates the top-down "Sonar-style" **Bird's Eye View** image (`/fusion/bev`) for operator awareness. |
| **`yolo3d_markers_node`** | **Visualization** | Publishes the **Green 3D Bounding Boxes** and text labels in RViz. |
| **`usb_cam`** | **Driver** | Interacts with the hardware camera (`/dev/video0`) to publish raw images. |
| **`image_converter_node`** | **Utility** | Converts raw YUYV camera images to BGR8 format to ensure compatibility with TensorRT and RTAB-Map. |

---

## 📐 TF Tree (Coordinate Frames)

Understanding the coordinate frames is crucial for navigation.

| Parent Frame | Child Frame | Publisher | Description |
| :--- | :--- | :--- | :--- |
| **`map`** | **`odom`** | `static_transform_publisher` | The global reference frame. Currently static (0,0,0) for local pool testing. |
| **`odom`** | **`base_link`** | `rtabmap_odom` (or static) | **The Critical Link.** Represents the robot moving in the pool. Visual Odometry publishes this update. |
| **`base_link`** | **`camera_link`** | `robot_state_publisher` | Defined in `URDF`. The mounting position of the camera relative to the robot's center. |
| **`camera_link`** | **`camera_link_optical`** | `robot_state_publisher` | Handles the rotation from standard ROS coordinates (X-forward) to Camera coordinates (Z-forward). |
| **`base_link`** | **`scan`** | `pointcloud_to_laserscan` | Virtual laser frame. Projects the 3D depth cloud into a flat 2D slice for simple obstacle avoidance. |

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **"System Throttled" Warning** | Jetson is drawing too much power (Over-current). | Run `sudo nvpmodel -m 1` to switch to 15W Mode. |
| **Robot Glitches / Teleports** | TF Conflict: You have two nodes publishing `odom` → `base_link`. | Disable `static_transform_publisher` in the launch file when running Visual Odometry. |
| **Map is Empty (No Blocks)** | Octomap is filtering out the pool floor as "ground". | Set `filter_ground: false` in `octomap_mapping.launch.py`. |
| **YOLO Detects Nothing** | Underwater color absorption is confusing the model. | Enable `preprocess_underwater` in `yolo11_trt_node.cpp` (CLAHE + White Balance). |
| **Low FPS (< 5)** | Input resolution is too high for the Jetson. | In `launch.py`, change `input_width` to `640` or reduce framerate to `5.0`. |
| **Objects in Wrong Place** | Camera intrinsics are calibrated for air, not water. | Multiply your air `fx` and `fy` parameters by **1.33** in `params.yaml`. |


## **🔮 Future Roadmap: Sensor Fusion**
To upgrade from Visual Odometry to a full Navigation Grade system, integrate the **Robot Localization** package:

* **Input 1**: Visual Odometry (X, Y Position)

* **Input 2**: DVL (X, Y Velocity)

* **Input 3**: IMU (Orientation / Gyro)

* **Input 4**: Depth Sensor (Z Position)