#include "yolo3d_stack/depth_anything_node.hpp"


DepthAnythingNode::DepthAnythingNode() : Node("depth_anything_node") {
auto it = image_transport::ImageTransport(shared_from_this());
std::string cam_topic = this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
std::string model = this->declare_parameter<std::string>("model_path", "");
in_w_ = this->declare_parameter<int>("input_width", 640);
in_h_ = this->declare_parameter<int>("input_height", 480);
use_cuda_ = this->declare_parameter<bool>("use_cuda", true);
std::string out_topic = this->declare_parameter<std::string>("output_topic", "/depth/map");


net_ = cv::dnn::readNet(model);
if (use_cuda_) {
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
} else {
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}


sub_ = it.subscribe(cam_topic, 1, &DepthAnythingNode::imageCb, this);
depth_pub_ = it.advertise(out_topic, 1);
}


void DepthAnythingNode::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
cv_bridge::CvImageConstPtr cv_ptr;
try { cv_ptr = cv_bridge::toCvShare(msg, "bgr8"); }
catch (...) { RCLCPP_ERROR(get_logger(), "cv_bridge fail"); return; }
cv::Mat frame = cv_ptr->image;


// Preprocess to model input size, normalization as per DA small export (usually [0,1], RGB)
cv::Mat rgb; cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
cv::Mat blob; cv::dnn::blobFromImage(rgb, blob, 1.0/255.0, cv::Size(in_w_, in_h_), cv::Scalar(), true, false);
net_.setInput(blob);
cv::Mat out = net_.forward(); // shape [1,1,H,W] predicted relative depth


// Resize back to original size
cv::Mat depth_small(out.size[2], out.size[3], CV_32FC1, out.ptr<float>());
cv::Mat depth;
cv::resize(depth_small, depth, frame.size(), 0, 0, cv::INTER_LINEAR);


// Optional: scale/normalize to approximate meters (monocular depth is relative). Here we min-max to [0,10] m as a heuristic; tune later or use metric DA variants.
double minv, maxv; cv::minMaxLoc(depth, &minv, &maxv);
cv::Mat depth_m = (depth - minv) / std::max(1e-6, (maxv-minv));
depth_m = 0.5 + 9.5 * depth_m; // 0.5m..10m heuristic


auto out_msg = cv_bridge::CvImage(msg->header, "32FC1", depth_m).toImageMsg();
depth_pub_.publish(out_msg);
}
