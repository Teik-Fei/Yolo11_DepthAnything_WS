# YOLO11 + Depth Anything + Fusion (ROS2 Humble)

ROS2 C++ stack to run YOLO11 (n) + Depth Anything (small) and fuse into 3D detections + BEV.
Developed in WSL2; deploy on Jetson Orin Nano/NX.

## Build
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

ros2 launch yolo3d_stack yolo3d_bringup.launch.py


Commit:
```bash
git add .
git commit -m "Initial ROS2 C++ YOLO11+DepthAnything+Fusion stack"
