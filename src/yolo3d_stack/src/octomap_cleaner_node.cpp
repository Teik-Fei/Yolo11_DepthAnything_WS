/**
 * OctoMap Persistent Cleaner Node
 * 
 * Purpose: Pass through detections to build persistent map
 * 
 * Key Improvements:
 * 1. Publishes ONLY the latest detection frame (not merged history)
 * 2. NO timeout clearing - objects are remembered
 * 3. Filters duplicate/low-quality detections
 * 4. OctoMap server handles persistence
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class OctoMapCleanerNode : public rclcpp::Node {
public:
    OctoMapCleanerNode() : Node("octomap_cleaner_node") {
        RCLCPP_INFO(this->get_logger(), "🗺️  OctoMap Persistent Cleaner - Building Memory");
        
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
        
        RCLCPP_INFO(this->get_logger(), 
            "Persistent mode enabled | Min points: %d",
            min_points_threshold_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clean_cloud_pub_;
    
    int min_points_threshold_;
    
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
        
        RCLCPP_DEBUG(this->get_logger(), 
            "✅ Published detection: %zu points from frame '%s'",
            pc.points.size(), msg->header.frame_id.c_str());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OctoMapCleanerNode>());
    rclcpp::shutdown();
    return 0;
}
