#ifndef OCTOMAP_CLEANER_NODE_HPP_
#define OCTOMAP_CLEANER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <deque>

/**
 * OctoMap Voxel Cleaner Node
 * 
 * Prevents voxel accumulation when moving objects (like poles) are detected.
 * Uses a sliding time window to keep only recent point clouds.
 * 
 * Subscribes: /yolo/obstacle_cloud (from YOLO detection)
 * Publishes: /octomap_clean/cloud_in (to OctoMap)
 * 
 * Configuration:
 * - max_cloud_age_ms: Maximum age of clouds to keep (default: 500ms)
 * - max_buffer_size: Maximum number of clouds in buffer (default: 5)
 */

class OctoMapCleanerNode : public rclcpp::Node {
public:
    explicit OctoMapCleanerNode(const rclcpp::NodeOptions & options);

private:
    struct CloudBuffer {
        rclcpp::Time timestamp;
        sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    };
    
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    std::deque<CloudBuffer> cloud_buffer_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clean_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;
    
    int max_cloud_age_ms_;
    int max_buffer_size_;
};

#endif // OCTOMAP_CLEANER_NODE_HPP_
