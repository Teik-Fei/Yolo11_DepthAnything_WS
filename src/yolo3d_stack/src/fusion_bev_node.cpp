#include "yolo3d_stack/fusion_bev_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <rclcpp/qos.hpp>

FusionBevNode::FusionBevNode(const rclcpp::NodeOptions & options) 
    : Node("fusion_bev_node", options), last_sync_time_(this->now()) // Initialize watchdog timer
{
    // QoS Settings
    rclcpp::QoS qos_best_effort = rclcpp::SensorDataQoS(); // Best Effort for Subscribers
    rclcpp::QoS qos_reliable(10); // Reliable for Publishers (RQT)

    // Parameters
    std::string img_topic = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    std::string depth_topic = this->declare_parameter<std::string>("depth_topic", "/depth/image_raw");
    std::string det_topic = this->declare_parameter<std::string>("detections_topic", "/yolo/detections"); 
    
    scale_factor_ = this->declare_parameter<float>("scale_factor", 0.036);
    cam_fx_ = this->declare_parameter<float>("camera_fx", 600.0);
    cam_cx_ = this->declare_parameter<float>("camera_cx", 320.0);

    RCLCPP_INFO(this->get_logger(), "Initializing Fusion BEV Node (Latest Data Mode)...");

    // Subscribe individually (QoS set to Best Effort to receive frames from usb_cam)
    sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
        img_topic, qos_best_effort, std::bind(&FusionBevNode::imgCb, this, std::placeholders::_1));

    sub_depth_ = this->create_subscription<sensor_msgs::msg::Image>(
        depth_topic, qos_best_effort, std::bind(&FusionBevNode::depthCb, this, std::placeholders::_1));

    sub_dets_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
        det_topic, qos_best_effort, std::bind(&FusionBevNode::detCb, this, std::placeholders::_1));

    // Publisher (Reliable QoS for RQT)
    bev_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/fusion/bev", qos_reliable);

    // Processing Timer (Runs independently at 30 Hz)
    timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&FusionBevNode::processTimerCb, this));
}

// Individual Callbacks (Simply cache the latest message)
void FusionBevNode::imgCb(const sensor_msgs::msg::Image::SharedPtr msg) { last_img_ = msg; }
void FusionBevNode::depthCb(const sensor_msgs::msg::Image::SharedPtr msg) { last_depth_ = msg; }
void FusionBevNode::detCb(const vision_msgs::msg::Detection2DArray::SharedPtr msg) { last_dets_ = msg; }


void FusionBevNode::processTimerCb() 
{
    // 1. Check if we have data from all sources
    bool img_ok = (last_img_ != nullptr);
    bool depth_ok = (last_depth_ != nullptr);
    bool dets_ok = (last_dets_ != nullptr);

    if (!img_ok || !depth_ok || !dets_ok) {
        // Watchdog warning
        static auto last_warn = this->now();
        if ((this->now() - last_warn).seconds() > 2.0) {
            RCLCPP_WARN(this->get_logger(), "Waiting for data... (Img: %s, Depth: %s, Dets: %s)", 
                img_ok ? "OK" : "NO", depth_ok ? "OK" : "NO", dets_ok ? "OK" : "NO");
            last_warn = this->now();
        }
        return;
    }
    
    // Reset watchdog now that we have data
    last_sync_time_ = this->now();
    
    // Throttle log to prove it's running
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
        "Fusion ACTIVE! Processing %ld detections.", last_dets_->detections.size());

    try {
        // We use the cached pointers to the latest data
        cv::Mat depth = cv_bridge::toCvCopy(last_depth_, "32FC1")->image;
        
        int img_w = 640; 
        int img_h = 640;
        cv::Mat bev = cv::Mat::zeros(img_h, img_w, CV_8UC3); // Black Canvas

        // Draw Grid & Text
        cv::line(bev, cv::Point(img_w/2, 0), cv::Point(img_w/2, img_h), cv::Scalar(100, 100, 100), 1);
        cv::line(bev, cv::Point(0, img_h - 100), cv::Point(img_w, img_h - 100), cv::Scalar(50, 50, 50), 1);
        cv::putText(bev, "BEV ACTIVE", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        for (const auto& det : last_dets_->detections) {
            float cx = det.bbox.center.position.x;
            float cy = det.bbox.center.position.y;
            int x = static_cast<int>(cx);
            int y = static_cast<int>(cy);

            if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows) continue;

            float d_raw = depth.at<float>(y, x);
            float z_m = d_raw * scale_factor_; 

            // Log the projected distance
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Det Z_m=%.2fm", z_m);

            if (z_m <= 0.05 || z_m > 40.0) continue; 

            float x_m = (x - cam_cx_) * z_m / cam_fx_;

            // BEV Mapping (100px per meter)
            int bev_x = (int)((x_m * 100) + img_w/2);
            int bev_y = (int)(img_h - (z_m * 100));

            if (bev_x < 0 || bev_x >= img_w || bev_y < 0 || bev_y >= img_h) continue;

            // Draw Dot
            cv::circle(bev, cv::Point(bev_x, bev_y), 10, cv::Scalar(0, 0, 255), -1);
            cv::putText(bev, std::to_string((int)z_m) + "m", cv::Point(bev_x+15, bev_y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        auto bev_msg = cv_bridge::CvImage(last_img_->header, "bgr8", bev).toImageMsg();
        bev_pub_->publish(*bev_msg);

    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(FusionBevNode)