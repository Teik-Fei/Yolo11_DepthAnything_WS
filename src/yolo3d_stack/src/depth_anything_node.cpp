#include "yolo3d_stack/depth_anything_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
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
    
    // --- FIX APPLIED HERE ---

    // Step 1: Prevent division by zero errors
    cv::max(depth, 0.1, depth); 

    // Step 2: INVERT the depth
    
    // Declare parameter if missing, default to 4.0
    if (!this->has_parameter("depth_factor")) {
        this->declare_parameter<float>("depth_factor", 4.0);
    }
    
    // Read the parameter from YAML (or use default)
    float depth_factor = this->get_parameter("depth_factor").as_double();

    // Perform Inversion: Real_Meters = Factor / Model_Output
    cv::divide(depth_factor, depth, depth);

    // --- FIX ENDS HERE ---

    // Publish...
    auto out_msg = cv_bridge::CvImage(msg->header, "32FC1", depth).toImageMsg();
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