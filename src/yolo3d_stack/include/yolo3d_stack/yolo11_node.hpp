#ifndef YOLO11_NODE_HPP_
#define YOLO11_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/vision_msgs/msg/detection2_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class Yolo11Node : public rclcpp::Node {
public:
    // CRITICAL: Declare constructor with options
    explicit Yolo11Node(const rclcpp::NodeOptions & options);

private:
    void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    cv::dnn::Net net_;
    float conf_thres_;
    bool use_cuda_;
    std::vector<std::string> classes_;
};

#endif // YOLO11_NODE_HPP_