/**
 * OctoMap Voxel Cleaner Node - Simple Version for 2D Costmap
 * 
 * Purpose: Keep only recent voxels, project to 2D costmap
 * 
 * This node:
 * 1. Subscribes to obstacle cloud from YOLO
 * 2. Removes old voxels (prevents accumulation)
 * 3. Publishes cleaned cloud to OctoMap
 * 4. OctoMap automatically projects to 2D costmap
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <deque>

class OctoMapCleanerNode : public rclcpp::Node {
public:
    OctoMapCleanerNode() : Node("octomap_cleaner_node") {
        RCLCPP_INFO(this->get_logger(), "🗺️  OctoMap Cleaner for 2D Costmap");
        
        // Parameters - tuned for costmap generation
        max_cloud_age_ms_ = this->declare_parameter<int>("max_cloud_age_ms", 1500);  // 1.5 sec window
        max_buffer_size_ = this->declare_parameter<int>("max_buffer_size", 10);      // Keep 10 frames
        
        // Subscribers & Publishers
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/yolo/obstacle_cloud",
            rclcpp::SensorDataQoS(),
            std::bind(&OctoMapCleanerNode::cloudCallback, this, std::placeholders::_1)
        );
        
        // Publish cleaned cloud to OctoMap
        clean_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/octomap_clean/cloud_in", 10
        );
        
        RCLCPP_INFO(this->get_logger(), 
            "Keeping clouds from last %d ms (max %d frames)",
            max_cloud_age_ms_, max_buffer_size_);
    }

private:
    struct CloudBuffer {
        rclcpp::Time timestamp;
        sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    };
    
    std::deque<CloudBuffer> cloud_buffer_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clean_cloud_pub_;
    
    int max_cloud_age_ms_;
    int max_buffer_size_;
    
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto now = this->now();
        
        // Add new cloud to buffer
        cloud_buffer_.push_back({now, msg});
        
        // Remove old clouds
        while (!cloud_buffer_.empty()) {
            auto age_ms = (now - cloud_buffer_.front().timestamp).total_milliseconds();
            if (age_ms > max_cloud_age_ms_ || cloud_buffer_.size() > (size_t)max_buffer_size_) {
                cloud_buffer_.pop_front();
            } else {
                break;
            }
        }
        
        // Merge all recent clouds
        pcl::PointCloud<pcl::PointXYZ> merged_cloud;
        for (const auto& buf : cloud_buffer_) {
            pcl::PointCloud<pcl::PointXYZ> pc;
            pcl::fromROSMsg(*buf.cloud, pc);
            merged_cloud += pc;
        }
        
        // Publish merged cloud
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(merged_cloud, output_msg);
        output_msg.header = msg->header;
        clean_cloud_pub_->publish(output_msg);
        
        RCLCPP_DEBUG(this->get_logger(), "Merged: %zu points from %zu buffers",
            merged_cloud.size(), cloud_buffer_.size());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OctoMapCleanerNode>());
    rclcpp::shutdown();
    return 0;
}
