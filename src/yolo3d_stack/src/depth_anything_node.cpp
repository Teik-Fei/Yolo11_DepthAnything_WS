// Code for running depth_anything_node.cpp using CPU
/*
#include "yolo3d_stack/depth_anything_node.hpp"
#include <rclcpp/qos.hpp>

DepthAnythingNode::DepthAnythingNode(const rclcpp::NodeOptions & options) 
    : Node("depth_anything_node", options) 
{
    // --- Parameters ---
    std::string cam_topic = this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
    std::string model_path = this->declare_parameter<std::string>("model_path", "/home/mecatron/depth_anything_v2_vits.onnx");
    std::string out_topic = this->declare_parameter<std::string>("output_topic", "/depth/image_raw");
    
    in_w_ = this->declare_parameter<int>("input_width", 518);
    in_h_ = this->declare_parameter<int>("input_height", 518);
    use_cuda_ = this->declare_parameter<bool>("use_cuda", true);

    RCLCPP_INFO(this->get_logger(), "Initializing Depth Anything Node...");

    // --- Load Network ---
    if (!model_path.empty()) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (!net_.empty()) {
                if (use_cuda_) {
                    RCLCPP_INFO(this->get_logger(), "Using CUDA (FP16)");
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                } else {
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
            }
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "No model path provided!");
    }

    // --- Subscribers & Publishers ---
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        cam_topic, 
        rclcpp::SensorDataQoS(), 
        std::bind(&DepthAnythingNode::imageCb, this, std::placeholders::_1));
        
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(out_topic, 10);
}

void DepthAnythingNode::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (net_.empty()) return;

    static cv::Mat last_depth;

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    if (cv_ptr->image.empty()) return;

    // Preprocessing
    cv::Mat blob;
    cv::dnn::blobFromImage(cv_ptr->image, blob, 1.0/255.0, cv::Size(in_w_, in_h_), cv::Scalar(0.485, 0.456, 0.406), true, false);
    
    net_.setInput(blob);
    
    cv::Mat out;
    try {
        out = net_.forward();
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", e.what());
        return;
    }

    if (out.empty()) return;

    // Handle output dimensions
    int out_h = 0;
    int out_w = 0;

    if (out.dims == 4) {
        out_h = out.size[2];
        out_w = out.size[3];
    } else if (out.dims == 3) {
        out_h = out.size[1];
        out_w = out.size[2];
    } else {
        return;
    }

    if (out_h <= 0 || out_w <= 0) return;

    // === Conversion to Depth ===
    
    cv::Mat depth_small(out_h, out_w, CV_32FC1, out.ptr<float>());
    cv::Mat depth;
    cv::resize(depth_small, depth, cv_ptr->image.size(), 0, 0, cv::INTER_LINEAR);

    // Step 1: Prevent division by zero errors
    cv::max(depth, 0.1, depth); 

    // Step 2: INVERT the depth
    float depth_factor = this->get_parameter("depth_factor").as_double();
    cv::divide(depth_factor, depth, depth);
    // ---------------------

    // --- NEW SMOOTHING LOGIC ---
    // Alpha controls smoothness. 
    // 0.1 = Very smooth (slow reaction), 1.0 = No smoothing (instant reaction).
    // Start with 0.3 for a good balance.
    float alpha = 0.3f; 

    static cv::Mat accumulated_depth; // Holds the "memory"

    if (accumulated_depth.empty() || accumulated_depth.size() != depth.size()) {
        accumulated_depth = depth.clone(); // First frame, just copy
    } else {
        // formula: current = (alpha * new) + ((1-alpha) * old)
        cv::addWeighted(depth, alpha, accumulated_depth, (1.0 - alpha), 0.0, accumulated_depth);
    }
    
    // Publish the SMOOTHED version, not the raw 'depth'
    auto out_msg = cv_bridge::CvImage(msg->header, "32FC1", accumulated_depth).toImageMsg();
    depth_pub_->publish(*out_msg);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DepthAnythingNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
} */

/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/

// Code for running depth_anything_node.cpp using GPU
#include "yolo3d_stack/depth_anything_node.hpp"
#include <rclcpp/qos.hpp>

DepthAnythingNode::DepthAnythingNode(const rclcpp::NodeOptions & options) 
    : Node("depth_anything_node", options) 
{
    // --- Parameters ---
    std::string cam_topic = this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
    std::string model_path = this->declare_parameter<std::string>("model_path", "/home/mecatron/depth_anything_v2_vits.onnx");
    std::string out_topic = this->declare_parameter<std::string>("output_topic", "/depth/image_raw");
    
    in_w_ = this->declare_parameter<int>("input_width", 518);
    in_h_ = this->declare_parameter<int>("input_height", 518);
    use_cuda_ = this->declare_parameter<bool>("use_cuda", true);

    this->declare_parameter<float>("max_depth_m", 5.0);
    this->declare_parameter<float>("scale_factor", 3.0);

    // Declare depth_factor here so it is ready when the callback runs
    this->declare_parameter<float>("depth_factor", 0.75);

    RCLCPP_INFO(this->get_logger(), "Initializing Depth Anything Node...");

    // --- Load Model ---
    if (!model_path.empty()) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (!net_.empty()) {
                if (use_cuda_) {
                    RCLCPP_INFO(this->get_logger(), "Using CUDA (FP16)");
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                } else {
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
            }
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "No model path provided!");
    }

    // --- Subscribers & Publishers ---
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        cam_topic, 
        rclcpp::SensorDataQoS(), 
        std::bind(&DepthAnythingNode::imageCb, this, std::placeholders::_1));
        
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(out_topic, 10);
}

void DepthAnythingNode::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (net_.empty()) return;

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (...) { return; }

    cv::Mat blob;
    cv::dnn::blobFromImage(
        cv_ptr->image, blob, 1.0/255.0, cv::Size(in_w_, in_h_),
        cv::Scalar(0.485, 0.456, 0.406), true, false
    );

    net_.setInput(blob);

    cv::Mat out = net_.forward();
    int out_h = 0;
    int out_w = 0;

    if (out.dims == 3) {
        // Shape = [1, H, W]
        out_h = out.size[1];
        out_w = out.size[2];
    }
    else if (out.dims == 4) {
        // Shape = [1, 1, H, W]
        out_h = out.size[2];
        out_w = out.size[3];
    }
    else {
        RCLCPP_ERROR(this->get_logger(), "Unexpected output dims: %d", out.dims);
        return;
    }
    if (out_h <= 1 || out_w <= 1) {
        RCLCPP_ERROR(this->get_logger(),
                 "Invalid output size: out_h=%d out_w=%d", out_h, out_w);
        return;
    }

    cv::Mat depth_small(out_h, out_w, CV_32FC1, out.ptr<float>());

    cv::Mat depth_inv = 1.0 / depth_small;

    // Get scale factor
    float scale_factor = this->get_parameter("scale_factor").as_double();
    cv::Mat depth_scaled = depth_inv * scale_factor;

    // Resize to match the original RGB camera resolution
    cv::Mat depth;
    cv::resize(depth_scaled, depth, cv_ptr->image.size(), 0, 0, cv::INTER_LINEAR);

    // Temporal smoothing
    static cv::Mat accumulated;
    float alpha = 0.4f;

    if (accumulated.empty() || accumulated.size() != depth.size()) {
        accumulated = depth.clone();
    } else {
        cv::addWeighted(depth, alpha, accumulated, 1.0 - alpha, 0.0, accumulated);
    }
    // Publish final depth (meters, approx)
    auto out_msg = cv_bridge::CvImage(msg->header, "32FC1", accumulated).toImageMsg();
    depth_pub_->publish(*out_msg);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DepthAnythingNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}