# YOLO11 + Depth Anything V2 + 3D Fusion (ROS 2 Humble)

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin-green)
![Acceleration](https://img.shields.io/badge/Acceleration-TensorRT%20%7C%20CUDA-orange)

A high-performance Perception Stack for ROS 2 developed for **Autonomous Underwater Vehicles (AUV)** but applicable to any robot.

This stack fuses **YOLO11 (TensorRT)** 2D object detection with **Depth Anything V2 (CUDA)** monocular depth estimation to create a real-time **3D Spatial Awareness** system. It outputs 3D bounding boxes, a dense PointCloud, and a top-down "Sonar-style" Bird's Eye View (BEV) map.

## ✨ Key Features
* **🚀 TensorRT Accelerated:** YOLO11 runs on the GPU using TensorRT 10.x for real-time inference.
* **🌊 Monocular Depth:** Uses "Depth Anything V2" (Small) via OpenCV CUDA to estimate distance from a single camera.
* **⚖️ Temporal Smoothing:** Implements Exponential Moving Average (EMA) filtering to stabilize depth jitter.
* **📡 AUV Radar/BEV Map:** Generates a top-down, distance-ringed view of detected objects (Sonar style) for navigation.
* **🧊 3D Visualization:** Projects 2D detections into 3D space for RViz markers and point clouds.

## 🛠️ Hardware & Software Requirements
* **Platform:** NVIDIA Jetson Orin Nano / Orin NX (or generic PC with NVIDIA GPU)
* **OS:** Ubuntu 22.04 (ROS 2 Humble)
* **Drivers:** CUDA 11.4+, TensorRT 8.6+ (JetPack 6.x recommended)
* **Camera:** USB Camera (e.g., Logitech C270) or CSI Camera

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    mkdir -p ~/yolo3d_ws/src
    cd ~/yolo3d_ws/src
    git clone <https://github.com/Teik-Fei/Yolo11_DepthAnything_WS.git>
    ```

2.  **Install Dependencies**
    ```bash
    sudo apt install ros-humble-vision-msgs ros-humble-image-transport ros-humble-cv-bridge ros-humble-usb-cam
    sudo pip3 install ultralytics onnxruntime-gpu
    ```

3.  **Model Preparation**
    * **YOLO11:** Export your model to TensorRT engine:
        ```bash
        yolo export model=yolo11n.pt format=engine half=True device=0
        ```
    * **Depth Anything:** Ensure you have the `.onnx` model (e.g., `depth_anything_v2_vits.onnx`).

4.  **Build**
    ```bash
    cd ~/yolo3d_ws
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select yolo3d_stack
    source install/setup.bash
    ```

## 🚀 Usage

**Launch the full stack (Camera + YOLO + Depth + Visualization):**
```bash
ros2 launch yolo3d_stack yolo3d_bringup.launch.py
```

### 🖥️ Visualization

The launch file automatically opens **RViz2** with a pre-configured view.

* **3D View:** Displays the dense PointCloud (colored by depth) and Green 3D Bounding Boxes around detected objects.
* **BEV Panel (Radar):** A top-down "Sonar-style" view on the topic `/fusion/bev`. This shows the robot at the bottom center and objects plotted on a distance grid.
* **Debug View:** A 2D camera feed with bounding boxes, class names, and estimated distances overlayed on the topic `/yolo/debug_image`.

### ⚙️ Configuration (`params.yaml`)

Configuration is managed in `src/yolo3d_stack/config/params.yaml`.

### 1. Depth Calibration (Critical)
The "Depth Anything" model provides relative depth. You **must** tune the `depth_factor` to match your specific camera and environment (water vs. air refraction).

**Calibration Steps:**
1.  Place an object exactly **2.0 meters** away from the camera.
2.  Check the distance shown in the Debug View or BEV map.
3.  Update the factor using: `New_Factor = Current_Factor * (Real_Dist / Seen_Dist)`

```yaml
depth_anything_node:
  ros__parameters:
    depth_factor: 4.0  # Increase if reading is too close, decrease if too far
    input_width: 518   # Model input size (do not change for Small model)
```

### 2. AUV Radar / BEV Settings   
Adjust the "Bird's Eye View" grid to change the scale of the map.
```yaml
fusion_bev_node:
  ros__parameters:
    meters_to_pixels: 100.0 # Scale: 100 pixels = 1 meter
    bev_size: 600           # Output image size: 600x600 pixels
```
### 📊 Performance Monitoring
To verify that YOLO and Depth models are running on the GPU (Hardware Acceleration), use `jtop` (part of `jetson-stats`).
**Installation**
```bash
sudo pip3 install -U jetson-stats
```
**Check Status**: Run `jtop` and switch to the **GPU** tab.

Look for processes `yolo11_trt_node` and `depth_anything_node`.

Ensure the Type column shows `G` (GPU).

If Type shows `C` (CPU), check your CUDA/TensorRT installation.

### 🧩 Nodes Description
| Node Name            | Function                                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------------------------|
| `usb_cam`             | Publishes the raw RGB camera feed from `/dev/video0`.                                                     |
| `yolo11_trt_node`     | Performs object detection using TensorRT. Publishes 2D boxes + computes 3D coordinates using depth map.    |
| `depth_anything_node` | Runs Depth Anything V2 using OpenCV CUDA and applies EMA filtering for stability.                         |
| `fusion_bev_node`     | Generates the Bird’s-Eye “Radar/Sonar” map from fused 3D detections.                                      |
| `depth_to_pointcloud`| Converts the depth image into a `sensor_msgs/PointCloud2` for rendering in RViz.                          |
| `yolo3d_markers`      | Publishes RViz Marker cubes to visualize 3D bounding boxes in real-time.                                  |







