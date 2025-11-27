/*#include "yolo3d_stack/fusion_bev_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp>  

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

## version 2

#include "yolo3d_stack/fusion_bev_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp>

FusionBevNode::FusionBevNode(const rclcpp::NodeOptions & options)
: Node("fusion_bev_node", options)
{
    // Params
    std::string det_topic = declare_parameter("detections_3d_topic", "/yolo3d/detections");
    meters_to_pixels_ = declare_parameter<float>("meters_to_pixels", 100.0f); 
    bev_size_         = declare_parameter<int>("bev_size", 600); 

    // Subscribe
    sub_dets_ = create_subscription<vision_msgs::msg::Detection3DArray>(
        det_topic, 10,
        std::bind(&FusionBevNode::detCallback, this, std::placeholders::_1)
    );

    // Publish
    bev_pub_ = create_publisher<sensor_msgs::msg::Image>("/fusion/bev", 10);

    // Timer (30 Hz)
    timer_ = create_wall_timer(std::chrono::milliseconds(33),
                               std::bind(&FusionBevNode::updateBev, this));
}

void FusionBevNode::detCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg){
    last_dets_ = msg;
}

void FusionBevNode::updateBev(){
    // 1. Create Canvas (Black)
    cv::Mat bev = cv::Mat::zeros(bev_size_, bev_size_, CV_8UC3);
    int cx = bev_size_ / 2;
    int cy = bev_size_; // Bottom of image is the robot

    // 2. Draw Radar Grid (Every 1 meter)
    for(int i=1; i<=5; i++) {
        int radius = i * meters_to_pixels_;
        cv::circle(bev, {cx, cy}, radius, {50, 50, 50}, 1); // Dark Grey rings
        cv::putText(bev, std::to_string(i)+"m", {cx + 5, cy - radius - 5}, 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {150,150,150}, 1);
    }
    cv::line(bev, {cx, 0}, {cx, bev_size_}, {50, 50, 50}, 1); // Center line

    // 3. Draw Objects
    if(last_dets_) {
        for(const auto &det : last_dets_->detections){
            if(det.results.empty()) continue;

            float X = det.bbox.center.position.x;
            float Z = det.bbox.center.position.z;

            // Convert Real World (Meters) -> Image (Pixels)
            // X is Left/Right. Z is Forward.
            float px = cx + (X * meters_to_pixels_);
            float py = cy - (Z * meters_to_pixels_);

            // Draw Dot (Red)
            if(px >=0 && px < bev_size_ && py >=0 && py < bev_size_) {
                cv::circle(bev, {int(px),int(py)}, 8, {0,0,255}, -1); 
                
                // Draw Label (White)
                std::string label = det.results[0].hypothesis.class_id;
                cv::putText(bev, label, {int(px)+10, int(py)}, 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 1);
            }
        }
    }

    // 4. Publish
    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    header.frame_id = "map"; // Static frame for visualization
    bev_pub_->publish(*cv_bridge::CvImage(header,"bgr8",bev).toImageMsg());
}

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FusionBevNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
} */



// Version 3
#include "yolo3d_stack/fusion_bev_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp>

//

FusionBevNode::FusionBevNode(const rclcpp::NodeOptions & options)
: Node("fusion_bev_node", options)
{
    // AUV Specific Params
    std::string det_topic = declare_parameter("detections_3d_topic", "/yolo3d/detections");
    meters_to_pixels_ = declare_parameter<float>("meters_to_pixels", 100.0f); // 1m = 100px
    bev_size_         = declare_parameter<int>("bev_size", 600); 

    sub_dets_ = create_subscription<vision_msgs::msg::Detection3DArray>(
        det_topic, 10,
        std::bind(&FusionBevNode::detCallback, this, std::placeholders::_1)
    );

    bev_pub_ = create_publisher<sensor_msgs::msg::Image>("/fusion/bev", 10);

    timer_ = create_wall_timer(std::chrono::milliseconds(33),
                               std::bind(&FusionBevNode::updateBev, this));
}

void FusionBevNode::detCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg){
    last_dets_ = msg;
}

void FusionBevNode::updateBev(){
    // 1. Create Dark "Sonar" Background
    cv::Mat bev = cv::Mat::zeros(bev_size_, bev_size_, CV_8UC3);
    
    // AUV is at the bottom center of the image
    int cx = bev_size_ / 2;
    int cy = bev_size_; 

    // 2. Draw Distance Rings (1m, 2m, 3m...)
    for(int i=1; i<=5; i++) {
        int radius = i * meters_to_pixels_;
        // Draw Arc
        cv::circle(bev, {cx, cy}, radius, {0, 100, 0}, 1); // Dark Green lines
        // Label Distance
        cv::putText(bev, std::to_string(i)+"m", {cx + 5, cy - radius - 5}, 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 150, 0}, 1);
    }
    
    // Draw Center Reference Line
    cv::line(bev, {cx, 0}, {cx, bev_size_}, {0, 50, 0}, 1); 

    // 3. Plot Detected Objects
    if(last_dets_) {
        for(const auto &det : last_dets_->detections){
            if(det.results.empty()) continue; // Skip if no name

            // Get Coordinates (Relative to Camera)
            float X = det.bbox.center.position.x; // Left/Right
            float Z = det.bbox.center.position.z; // Forward Depth

            // Convert to Pixels
            float px = cx + (X * meters_to_pixels_);
            float py = cy - (Z * meters_to_pixels_);

            // Draw Object if within range
            if(px >=0 && px < bev_size_ && py >=0 && py < bev_size_) {
                // Draw Bright Red Dot
                cv::circle(bev, {int(px),int(py)}, 6, {0,0,255}, -1); 
                
                // Draw Label (e.g. "Valve")
                std::string label = det.results[0].hypothesis.class_id;
                cv::putText(bev, label, {int(px)+10, int(py)}, 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
            }
        }
    }

    // 4. Publish
    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    header.frame_id = "map"; 
    bev_pub_->publish(*cv_bridge::CvImage(header,"bgr8",bev).toImageMsg());
}

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FusionBevNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}