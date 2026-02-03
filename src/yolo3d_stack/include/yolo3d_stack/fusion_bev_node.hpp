#ifndef FUSION_BEV_NODE_HPP_
#define FUSION_BEV_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class FusionBevNode : public rclcpp::Node {
public:
    explicit FusionBevNode(const rclcpp::NodeOptions & options);

private:
    // === Callbacks ===
    void detCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    void updateBev(); // timer loop

    // === Subscribers ===
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_dets_;

    // === Publisher ===
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bev_pub_;

    // === Storage ===
    vision_msgs::msg::Detection3DArray::SharedPtr last_dets_;

    // === BEV parameters ===
    float meters_to_pixels_;     // 1m -> px
    int   bev_size_;             // output image size

    // === Timer ===
    rclcpp::TimerBase::SharedPtr timer_;
};

#endif // FUSION_BEV_NODE_HPP_