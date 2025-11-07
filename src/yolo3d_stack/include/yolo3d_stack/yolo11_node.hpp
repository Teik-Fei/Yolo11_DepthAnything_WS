#pragma once
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


class Yolo11Node : public rclcpp::Node {
public:
Yolo11Node();
private:
void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
void loadModel();
void loadClasses(const std::string &path);


image_transport::Subscriber sub_;
rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
image_transport::Publisher vis_pub_;


cv::dnn::Net net_;
std::vector<std::string> classes_;
int in_w_, in_h_;
float conf_th_, iou_th_;
bool publish_vis_ = true;
bool use_cuda_ = true;
};
