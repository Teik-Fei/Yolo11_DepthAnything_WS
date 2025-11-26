#ifndef DEPTH_ANYTHING_NODE_HPP_
#define DEPTH_ANYTHING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class DepthAnythingNode : public rclcpp::Node {
public:
    explicit DepthAnythingNode(const rclcpp::NodeOptions & options);

private:
    void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;

    int in_w_;
    int in_h_;
    bool use_cuda_;
    cv::dnn::Net net_;
};

#endif // DEPTH_ANYTHING_NODE_HPP_