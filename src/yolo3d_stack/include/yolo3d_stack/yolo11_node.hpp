#ifndef YOLO11_NODE_HPP_
#define YOLO11_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <mutex>
#include <vector>
#include <string>

class Yolo11Node : public rclcpp::Node {
public:
    explicit Yolo11Node(const rclcpp::NodeOptions & options);

private:
    // Callbacks
    void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det3d_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    // Network + Labels
    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    float conf_thres_;

    // Depth buffer
    cv::Mat depth_img_;
    std::mutex depth_mutex_;

    // Camera intrinsic parameters
    float fx_, fy_, cx_, cy_;
};

#endif // YOLO11_NODE_HPP_