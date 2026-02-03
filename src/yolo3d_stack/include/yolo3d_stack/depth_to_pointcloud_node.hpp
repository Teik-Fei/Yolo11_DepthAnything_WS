#ifndef DEPTH_TO_POINTCLOUD_NODE_HPP_
#define DEPTH_TO_POINTCLOUD_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class DepthToPointcloudNode : public rclcpp::Node {
public:
    explicit DepthToPointcloudNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:

    // === Callback for synchronized depth + rgb ===
    void callback(const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg);

    // === Parameters ===
    std::string depth_topic_;
    std::string rgb_topic_;
    std::string cloud_topic_;

    double fx_, fy_, cx_, cy_;
    int step_;

    // === Publisher ===
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    // === Subscribers (message_filters) ===
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;

    // === Synchronizer ===
    std::shared_ptr<
        message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Image,
                sensor_msgs::msg::Image>>> sync_;
};

#endif
