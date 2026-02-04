/**
 * OctoMap Smart Cleaner Node - Prevents Growing Blocks
 * 
 * Purpose: Only publish CURRENT detection, not accumulated history
 * 
 * Key Improvements:
 * 1. Publishes ONLY the latest detection frame (not merged history)
 * 2. Implements automatic timeout clearing
 * 3. Filters duplicate/low-quality detections
 * 4. Prevents the "growing trail" effect when object moves
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class OctoMapCleanerNode : public rclcpp::Node {
public:
    OctoMapCleanerNode() : Node("octomap_cleaner_node") {
        RCLCPP_INFO(this->get_logger(), "🗺️  OctoMap Smart Cleaner - Prevents Growing Blocks");
        
        // ===== CRITICAL PARAMETER: Detection Timeout =====
        // If no detection received for this duration, publish EMPTY cloud to clear OctoMap
        detection_timeout_ms_ = this->declare_parameter<int>("detection_timeout_ms", 2000);  // 2 seconds
        
        // Minimum points to consider a valid detection (filter noise)
        min_points_threshold_ = this->declare_parameter<int>("min_points_threshold", 50);
        
        // Subscribers & Publishers
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/yolo/obstacle_cloud",
            rclcpp::SensorDataQoS(),
            std::bind(&OctoMapCleanerNode::cloudCallback, this, std::placeholders::_1)
        );
        
        // Publish cleaned cloud to OctoMap (ONLY latest frame, no history)
        clean_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/octomap_clean/cloud_in", 10
        );
        
        // Timeout timer to periodically clear stale detections
        timeout_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),  // Check every 500ms
            std::bind(&OctoMapCleanerNode::checkTimeout, this)
        );
        
        last_detection_time_ = this->now();
        
        RCLCPP_INFO(this->get_logger(), 
            "Detection timeout: %d ms | Min points: %d",
            detection_timeout_ms_, min_points_threshold_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clean_cloud_pub_;
    rclcpp::TimerBase::SharedPtr timeout_timer_;
    
    int detection_timeout_ms_;
    int min_points_threshold_;
    rclcpp::Time last_detection_time_;
    std::string last_frame_id_;
    
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert to PCL for analysis
        pcl::PointCloud<pcl::PointXYZ> pc;
        pcl::fromROSMsg(*msg, pc);
        
        // ===== FILTER 1: Empty cloud check =====
        if (pc.points.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Received empty detection cloud - no objects detected");
            return;
        }
        
        // ===== FILTER 2: Minimum points threshold =====
        if ((int)pc.points.size() < min_points_threshold_) {
            RCLCPP_DEBUG(this->get_logger(),
                "Filtered: Only %zu points (threshold: %d)", 
                pc.points.size(), min_points_threshold_);
            return;
        }
        
        // ===== PUBLISH ONLY CURRENT FRAME (NO ACCUMULATION) =====
        // This is the key fix: don't merge with history, just publish fresh detection
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(pc, output_msg);
        output_msg.header = msg->header;
        
        clean_cloud_pub_->publish(output_msg);
        
        // Update timestamp for timeout detection
        last_detection_time_ = this->now();
        last_frame_id_ = msg->header.frame_id;
        
        RCLCPP_DEBUG(this->get_logger(), 
            "✅ Published fresh detection: %zu points from frame '%s'",
            pc.points.size(), msg->header.frame_id.c_str());
    }
    
    void checkTimeout() {
        auto now = this->now();
        auto time_since_last_detection = (now - last_detection_time_).seconds() * 1000.0;
        
        // If no detection received for timeout duration, clear OctoMap
        if (time_since_last_detection > detection_timeout_ms_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Detection timeout (%.1f ms > %d ms). Publishing EMPTY cloud to clear OctoMap.",
                time_since_last_detection, detection_timeout_ms_);
            
            // Publish empty cloud to tell OctoMap "nothing here anymore"
            pcl::PointCloud<pcl::PointXYZ> empty_pcl;
            sensor_msgs::msg::PointCloud2 empty_cloud;
            pcl::toROSMsg(empty_pcl, empty_cloud);
            empty_cloud.header.stamp = now;
            empty_cloud.header.frame_id = last_frame_id_.empty() ? "base_link" : last_frame_id_;
            
            clean_cloud_pub_->publish(empty_cloud);
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OctoMapCleanerNode>());
    rclcpp::shutdown();
    return 0;
}
