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