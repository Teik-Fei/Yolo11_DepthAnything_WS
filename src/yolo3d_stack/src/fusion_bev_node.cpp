#include "yolo3d_stack/fusion_bev_node.hpp"
using vision_msgs::msg::Detection2DArray; using vision_msgs::msg::Detection2D;
using vision_msgs::msg::Detection3DArray; using vision_msgs::msg::Detection3D;
using vision_msgs::msg::ObjectHypothesisWithPose;


static inline float clampf(float v, float lo, float hi){return std::max(lo, std::min(v, hi));}


FusionBevNode::FusionBevNode() : Node("fusion_bev_node") {
// Camera intrinsics or derive from FOVs
fx_ = this->declare_parameter<double>("fx", 0.0);
fy_ = this->declare_parameter<double>("fy", 0.0);
cx_ = this->declare_parameter<double>("cx", 0.0);
cy_ = this->declare_parameter<double>("cy", 0.0);
fovx_deg_ = this->declare_parameter<double>("fov_x_deg", 60.0);
fovy_deg_ = this->declare_parameter<double>("fov_y_deg", 45.0);
img_w_ = this->declare_parameter<int>("img_width", 640);
img_h_ = this->declare_parameter<int>("img_height", 480);


if (fx_ == 0 || fy_ == 0) {
// derive focal from FOV: fx = (w/2)/tan(FOVx/2)
fx_ = (img_w_/2.0f) / std::tan((fovx_deg_*M_PI/180.0f)/2.0f);
fy_ = (img_h_/2.0f) / std::tan((fovy_deg_*M_PI/180.0f)/2.0f);
}
if (cx_ == 0) cx_ = img_w_/2.0f;
if (cy_ == 0) cy_ = img_h_/2.0f;


bev_forward_m_ = this->declare_parameter<double>("bev_forward_m", 20.0);
bev_side_m_ = this->declare_parameter<double>("bev_side_m", 10.0);
bev_res_m_ = this->declare_parameter<double>("bev_resolution_m", 0.1);
median_samples_ = this->declare_parameter<int>("median_depth_samples", 25);


std::string det_topic = this->declare_parameter<std::string>("det_topic", "/yolo/detections_2d");
std::string depth_topic = this->declare_parameter<std::string>("depth_topic", "/depth/map");
std::string det3d_topic = this->declare_parameter<std::string>("det3d_topic", "/yolo3d/detections_3d");
std::string bev_topic = this->declare_parameter<std::string>("bev_topic", "/yolo3d/bev");


det3d_pub_ = this->create_publisher<Detection3DArray>(det3d_topic, 10);
auto it = image_transport::ImageTransport(shared_from_this());
bev_pub_ = it.advertise(bev_topic, 1);


det_sub_ = std::make_shared<message_filters::Subscriber<Detection2DArray>>(this, det_topic);
depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, depth_topic);
sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *det_sub_, *depth_sub_);
sync_->registerCallback(std::bind(&FusionBevNode::cb, this, std::placeholders::_1, std::placeholders::_2));
}


void FusionBevNode::cb(const Detection2DArray::ConstSharedPtr &det,
const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg) {
cv_bridge::CvImageConstPtr depth_ptr;
try { depth_ptr = cv_bridge::toCvShare(depth_msg); }
catch (...) { RCLCPP_ERROR(get_logger(), "cv_bridge depth fail"); return; }
cv::Mat depth = depth_ptr->image; // 32FC1 meters approx


Detection3DArray arr3d; arr3d.header = det->header;


// Build BEV canvas
int bev_w = (int)(2*bev_side_m_/bev_res_m_);
int bev_h = (int)(bev_forward_m_/bev_res_m_);
}
