#ifndef FUSION_BEV_NODE_HPP_
#define FUSION_BEV_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/vision_msgs/msg/detection2_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class FusionBevNode : public rclcpp::Node {
public:
    explicit FusionBevNode(const rclcpp::NodeOptions & options);

private:
    // Individual Callbacks (Store the latest data)
    void imgCb(const sensor_msgs::msg::Image::SharedPtr msg);
    void depthCb(const sensor_msgs::msg::Image::SharedPtr msg);
    void detCb(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

    // Processing loop (Replaces message_filters::Synchronizer)
    void processTimerCb();

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_dets_;

    // Publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bev_pub_;

    // Data Cache (Latest messages received)
    sensor_msgs::msg::Image::SharedPtr last_img_;
    sensor_msgs::msg::Image::SharedPtr last_depth_;
    vision_msgs::msg::Detection2DArray::SharedPtr last_dets_;

    // Timer for processing and Watchdog
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time last_sync_time_; // Used for watchdog check

    // Params
    float scale_factor_;
    float cam_fx_, cam_cx_;
};

#endif // FUSION_BEV_NODE_HPP_