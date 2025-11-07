#include "yolo3d_stack/yolo11_node.hpp"
using vision_msgs::msg::Detection2DArray;
using vision_msgs::msg::Detection2D;
using vision_msgs::msg::ObjectHypothesisWithPose;


Yolo11Node::Yolo11Node() : Node("yolo11_node") {
auto it = image_transport::ImageTransport(shared_from_this());
std::string cam_topic = this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
std::string model = this->declare_parameter<std::string>("model_path", "");
std::string classes = this->declare_parameter<std::string>("classes_path", "");
in_w_ = this->declare_parameter<int>("input_width", 640);
in_h_ = this->declare_parameter<int>("input_height", 480);
conf_th_ = this->declare_parameter<double>("conf_threshold", 0.25);
iou_th_ = this->declare_parameter<double>("iou_threshold", 0.45);
publish_vis_ = this->declare_parameter<bool>("publish_visualized", true);
use_cuda_ = this->declare_parameter<bool>("use_cuda", true);


loadClasses(classes);


net_ = cv::dnn::readNet(model);
if (use_cuda_) {
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
} else {
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}


sub_ = it.subscribe(cam_topic, 1, &Yolo11Node::imageCb, this);
det_pub_ = this->create_publisher<Detection2DArray>("/yolo/detections_2d", 10);
vis_pub_ = it.advertise("/yolo/vis", 1);
}


void Yolo11Node::loadClasses(const std::string &path) {
classes_.clear();
if (path.empty()) return;
std::ifstream f(path);
std::string line;
while (std::getline(f, line)) if (!line.empty()) classes_.push_back(line);
}


void Yolo11Node::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
cv_bridge::CvImageConstPtr cv_ptr;
try { cv_ptr = cv_bridge::toCvShare(msg, "bgr8"); }
catch (...) { RCLCPP_ERROR(get_logger(), "cv_bridge fail"); return; }
cv::Mat frame = cv_ptr->image;


// Preprocess (YOLO11 default letterbox)
}
