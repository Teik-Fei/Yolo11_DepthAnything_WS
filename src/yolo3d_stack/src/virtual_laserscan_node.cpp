#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <cmath>

class VirtualLaserScanNode : public rclcpp::Node
{
public:
    VirtualLaserScanNode() : Node("virtual_laserscan_node")
    {
        // === Parameters ===
        // Topic to subscribe to (output of Depth Anything)
        depth_topic_ = declare_parameter("depth_topic", "/depth/image_raw");
        
        // Output Topic (Standard ROS name for Lidar)
        scan_topic_  = declare_parameter("scan_topic", "/scan");

        // Camera Intrinsics (Must match your Yolo/Depth nodes)
        fx_ = declare_parameter("fx", 555.715735370142);
        cx_ = declare_parameter("cx", 346.7216404016699);
        
        // Scan Settings
        // "scan_height": Number of vertical pixels to average/min-pool
        // 1 = Single line (noisy), 10 = 10 pixel band (safer)
        scan_height_ = declare_parameter("scan_height", 10); 
        
        // "scan_line_idx": Which vertical row to slice? 
        // -1 = Center of image (Standard)
        scan_line_idx_ = declare_parameter("scan_line_idx", -1);

        min_range_ = declare_parameter("min_range", 0.1);
        max_range_ = declare_parameter("max_range", 20.0);

        // Subscribers & Publishers
        auto qos = rclcpp::SensorDataQoS();
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            depth_topic_, qos,
            std::bind(&VirtualLaserScanNode::depthCb, this, std::placeholders::_1));

        pub_ = create_publisher<sensor_msgs::msg::LaserScan>(scan_topic_, 10);

        RCLCPP_INFO(get_logger(), "🟢 Virtual LaserScan Node Started");
    }

private:
    void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        // 1. Convert ROS Image to OpenCV
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        const cv::Mat& depth_img = cv_ptr->image;
        int height = depth_img.rows;
        int width  = depth_img.cols;

        // 2. Determine Scan Strip
        // If scan_line_idx is -1, use the vertical center
        int center_row = (scan_line_idx_ == -1) ? (height / 2) : scan_line_idx_;
        
        // Define top and bottom of the strip
        int row_start = std::max(0, center_row - (scan_height_ / 2));
        int row_end   = std::min(height, center_row + (scan_height_ / 2));

        // 3. Prepare LaserScan Message
        sensor_msgs::msg::LaserScan scan;
        scan.header = msg->header; // Keep timestamp and frame_id
        
        // Calculate Angles
        // Field of View Calculation based on Intrinsics
        // angle = atan( (x - cx) / fx )
        float angle_min = atan2(0 - cx_, fx_);           // Leftmost pixel
        float angle_max = atan2(width - 1 - cx_, fx_);   // Rightmost pixel
        
        scan.angle_min = angle_min;
        scan.angle_max = angle_max;
        scan.angle_increment = (angle_max - angle_min) / (width - 1);
        
        // Timing (Optional, but good for SLAM)
        scan.time_increment = 0.0;
        scan.scan_time = 0.033; // ~30 FPS
        scan.range_min = min_range_;
        scan.range_max = max_range_;

        scan.ranges.resize(width);

        // 4. Process Each Column to find Distance
        for (int u = 0; u < width; ++u) {
            
            // Find the closest obstacle in this vertical strip (Safety conservative)
            float min_depth_in_strip = 999.0f;
            
            for (int v = row_start; v < row_end; ++v) {
                float z = depth_img.at<float>(v, u);
                
                // Filter invalid pixels
                if (std::isfinite(z) && z > min_range_ && z < max_range_) {
                    if (z < min_depth_in_strip) {
                        min_depth_in_strip = z;
                    }
                }
            }

            // 5. Convert Depth (Z) to Range (R)
            // Lidar measures Euclidian distance (R), Depth camera measures Planar Z.
            // R = Z / cos(angle)
            if (min_depth_in_strip < 999.0f) {
                float angle = scan.angle_min + u * scan.angle_increment;
                float range = min_depth_in_strip / std::cos(angle);
                
                // Clamp max range
                if(range > max_range_) range = std::numeric_limits<float>::infinity();
                
                scan.ranges[u] = range;
            } else {
                scan.ranges[u] = std::numeric_limits<float>::infinity();
            }
        }

        pub_->publish(scan);
    }

    std::string depth_topic_, scan_topic_;
    float fx_, cx_;
    int scan_height_, scan_line_idx_;
    float min_range_, max_range_;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VirtualLaserScanNode>());
    rclcpp::shutdown();
    return 0;
}