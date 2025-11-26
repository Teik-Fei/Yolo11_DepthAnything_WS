#include "yolo3d_stack/fusion_bev_node.hpp"
//#include "rclcpp_components/register_node_macro.hpp"
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp>  // <-- ADD THIS

FusionBevNode::FusionBevNode(const rclcpp::NodeOptions & options)
: Node("fusion_bev_node", options)
{
    // === Parameters ===
    std::string det_topic = declare_parameter("detections_3d_topic", "/yolo3d/detections");
    meters_to_pixels_ = declare_parameter<float>("meters_to_pixels", 100.0f);    // scale = 1m → 100px
    bev_size_         = declare_parameter<int>("bev_size", 700);               // output resolution

    // === Subscribe YOLO3D ===
    sub_dets_ = create_subscription<vision_msgs::msg::Detection3DArray>(
        det_topic, rclcpp::SensorDataQoS(),
        std::bind(&FusionBevNode::detCallback, this, std::placeholders::_1)
    );

    // === Publish BEV Image ===
    bev_pub_ = create_publisher<sensor_msgs::msg::Image>("/fusion/bev", 10);

    // Run at 30Hz
    timer_ = create_wall_timer(std::chrono::milliseconds(33),
                               std::bind(&FusionBevNode::updateBev, this));

    RCLCPP_INFO(get_logger(), "Fusion BEV Node [3D Mode ✔]");
}


// Store last detections
void FusionBevNode::detCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg){
    last_dets_ = msg;
}


void FusionBevNode::updateBev(){
    if(!last_dets_) return;

    // === Create BEV Canvas ===
    cv::Mat bev = cv::Mat::zeros(bev_size_, bev_size_, CV_8UC3);

    int cx = bev_size_ / 2; // center = robot location

    cv::putText(bev,"BEV ACTIVE",{20,40},0,0.9,{0,255,0},2);
    cv::line(bev,{cx,0},{cx,bev_size_},{80,80,80},1); // center reference

    for(const auto &det : last_dets_->detections){
        if(det.results.empty()) continue;

        float X = det.bbox.center.position.x;   // left / right
        float Z = det.bbox.center.position.z;   // forward distance

        if(Z<=0.05 || Z>25) continue; // ignore far or invalid

        // Convert meters → pixels
        float px = cx + X * meters_to_pixels_;
        float py = bev_size_ - Z * meters_to_pixels_;

        if(px < 0 || px >= bev_size_ || py < 0 || py >= bev_size_) continue;

        cv::circle(bev, {int(px),int(py)}, 8, {0,0,255}, -1);

        std::string label = det.results[0].hypothesis.class_id +
                            " " + std::to_string((int)(Z))+"m";

        cv::putText(bev,label,{int(px)+12,int(py)},
                    0,0.55,{255,255,255},2);
    }

    bev_pub_->publish(*cv_bridge::CvImage(
        std_msgs::msg::Header(),"bgr8",bev).toImageMsg());
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FusionBevNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}