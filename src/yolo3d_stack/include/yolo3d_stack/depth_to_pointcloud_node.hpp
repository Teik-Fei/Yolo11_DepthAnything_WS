#ifndef DEPTH_TO_POINTCLOUD_NODE_HPP_
#define DEPTH_TO_POINTCLOUD_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

class DepthToPointcloudNode : public rclcpp::Node {
public:
    // Keep default arguments here to match your .cpp main function
    explicit DepthToPointcloudNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    void callback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    std::string depth_topic_, cloud_topic_;
    float fx_, fy_, cx_, cy_;
    int step_;
};

#endif // DEPTH_TO_POINTCLOUD_NODE_HPP_