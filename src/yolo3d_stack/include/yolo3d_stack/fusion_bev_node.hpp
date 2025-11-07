#pragma once
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>


class FusionBevNode : public rclcpp::Node {
public:
FusionBevNode();
private:
void cb(const vision_msgs::msg::Detection2DArray::ConstSharedPtr &det,
const sensor_msgs::msg::Image::ConstSharedPtr &depth);


// Params / camera intrinsics
float fx_, fy_, cx_, cy_;
float fovx_deg_, fovy_deg_;
int img_w_, img_h_;


// BEV params
float bev_forward_m_, bev_side_m_, bev_res_m_;
int median_samples_;


// pubs/subs
rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det3d_pub_;
image_transport::Publisher bev_pub_;


using SyncPolicy = message_filters::sync_policies::ApproximateTime<vision_msgs::msg::Detection2DArray, sensor_msgs::msg::Image>;
std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>> det_sub_;
std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};
