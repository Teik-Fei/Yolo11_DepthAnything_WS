#include "yolo3d_stack/depth_anything_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <rclcpp/qos.hpp>

DepthAnythingNode::DepthAnythingNode(const rclcpp::NodeOptions & options) 
    : Node("depth_anything_node", options) 
{
    // --- QoS SETTINGS (Reliable for RQT) ---
    rclcpp::QoS qos_profile(10);

    std::string cam_topic = this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
    std::string model_path = this->declare_parameter<std::string>("model_path", "");
    std::string out_topic = this->declare_parameter<std::string>("output_topic", "/depth/map");
    
    in_w_ = this->declare_parameter<int>("input_width", 518);
    in_h_ = this->declare_parameter<int>("input_height", 518);
    use_cuda_ = this->declare_parameter<bool>("use_cuda", true);

    RCLCPP_INFO(this->get_logger(), "Initializing Depth Node...");

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

    rmw_qos_profile_t qos = rmw_qos_profile_default;
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        cam_topic, 
        rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos), qos), 
        std::bind(&DepthAnythingNode::imageCb, this, std::placeholders::_1));
        
    // Publisher (Explicit Reliable QoS)
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(out_topic, qos_profile);
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

    cv::Mat depth_small(out_h, out_w, CV_32FC1, out.ptr<float>());
    
    cv::Mat depth;
    cv::resize(depth_small, depth, cv_ptr->image.size(), 0, 0, cv::INTER_LINEAR);

    double minv, maxv;
    cv::minMaxLoc(depth, &minv, &maxv);
    double range = std::max(1e-6, (maxv - minv));
    
    cv::Mat depth_m = (depth - minv) / range;
    depth_m = 0.5 + 9.5 * depth_m; 

    auto out_msg = cv_bridge::CvImage(msg->header, "32FC1", depth_m).toImageMsg();
    depth_pub_->publish(*out_msg);
}

RCLCPP_COMPONENTS_REGISTER_NODE(DepthAnythingNode)