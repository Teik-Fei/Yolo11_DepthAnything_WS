# OctoMap Growing Blocks Fix - Detailed Explanation

## The Problem ❌
When you move a detected object, OctoMap's blocks **grow in width** instead of just following the object's actual size. This creates a "trail" effect behind the object as it moves.

### Root Cause:
The old `octomap_cleaner_node` was **merging multiple detection frames into one accumulated cloud**:
```cpp
// OLD BEHAVIOR (Wrong):
for (const auto& buf : cloud_buffer_) {
    pcl::PointCloud<pcl::PointXYZ> pc;
    pcl::fromROSMsg(*buf.cloud, pc);
    merged_cloud += pc;  // ⚠️ ACCUMULATES HISTORY!
}
```

This meant if you detected the same object at positions [1, 1] and [2, 1], OctoMap would mark **both** positions as occupied, creating a trail.

---

## The Solution ✅

### Key Changes:

#### 1. **Publish ONLY Current Frame (No Accumulation)**
```cpp
// NEW BEHAVIOR (Correct):
sensor_msgs::msg::PointCloud2 output_msg;
pcl::toROSMsg(pc, output_msg);  // Just publish fresh detection
clean_cloud_pub_->publish(output_msg);
```

Now each frame is published independently. OctoMap only sees the **current** object position, not historical positions.

#### 2. **Automatic Timeout Clearing**
```cpp
// If no detection for 2 seconds, publish EMPTY cloud
if (time_since_last_detection > detection_timeout_ms_) {
    sensor_msgs::msg::PointCloud2 empty_cloud;
    // ... set empty cloud with proper header ...
    clean_cloud_pub_->publish(empty_cloud);  // Tells OctoMap to clear
}
```

This prevents stale detections from lingering forever in OctoMap.

#### 3. **Noise Filtering**
```cpp
// Only publish detections with enough points
if ((int)pc.points.size() < min_points_threshold_) {
    return;  // Skip small/noisy detections
}
```

---

## Configuration

### Launch File Parameters:
```python
Node(
    package='yolo3d_stack',
    executable='cloud_cleaner_node',
    name='cloud_cleaner',
    parameters=[{
        'detection_timeout_ms': 2000,      # How long before clearing stale detections
        'min_points_threshold': 50         # Minimum points for valid detection
    }]
)
```

### Tuning Guide:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `detection_timeout_ms` | 2000 | Clear OctoMap if no detection for 2 seconds |
| `min_points_threshold` | 50 | Ignore detections with <50 points (noise) |

**Adjust if:**
- **Too many false trails?** → Increase `detection_timeout_ms` to 3000-5000ms
- **Blocks disappear too fast?** → Decrease `detection_timeout_ms` to 1000ms
- **Too much noise?** → Increase `min_points_threshold` to 100-150

---

## What Changed in the Code:

### Old File (octomap_cleaner_node.cpp):
- ❌ Used `std::deque<CloudBuffer>` to store multiple frames
- ❌ Merged all frames into one cloud
- ❌ No timeout clearing mechanism
- ❌ Created growing trails as objects moved

### New File (octomap_cleaner_node.cpp):
- ✅ Publishes only the current detection frame
- ✅ Implements 500ms timeout timer to check for stale detections
- ✅ Automatically publishes empty clouds to clear OctoMap
- ✅ Filters low-point-count detections as noise
- ✅ Blocks maintain object-size proportions

---

## Expected Behavior:

### Before (Wrong):
```
Frame 1: Detect box at X=1  → OctoMap marks [1]
Frame 2: Detect box at X=2  → OctoMap marks [1,2] ← Trail grows!
Frame 3: Detect box at X=3  → OctoMap marks [1,2,3] ← Still growing!
```

### After (Correct):
```
Frame 1: Detect box at X=1  → OctoMap marks [1]
Frame 2: Detect box at X=2  → OctoMap clears [1], marks [2] ← Box follows movement!
Frame 3: Detect box at X=3  → OctoMap clears [2], marks [3] ← Clean update!
Frame 4: No detection       → OctoMap clears [3] after 2s timeout
```

---

## Rebuild & Test:

```bash
cd ~/yolo3d_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch yolo3d_stack yolo3d_bringup.launch.py
```

In RViz:
1. Add **OctoMap** visualization
2. Move the detected object around
3. ✅ Blocks should now **follow** the object without growing trails

---

## If Still Having Issues:

1. **Check message frequency**: `ros2 topic hz /yolo/obstacle_cloud`
   - If <5Hz, detections are too slow → increase detection_timeout_ms

2. **Check point cloud validity**: `ros2 topic echo /octomap_clean/cloud_in`
   - Should show fresh frame indices, not accumulated points

3. **Check OctoMap parameters** in `nav2_params.yaml`:
   - `voxel_size`: Larger values (0.1m) = smoother but less precise
   - `max_range`: Must match camera range

---

## Summary:
The fix changes the strategy from **"accumulate all detections"** to **"publish only current detection + auto-clear stale ones"**. This prevents the growing block effect while maintaining proper obstacle avoidance.
