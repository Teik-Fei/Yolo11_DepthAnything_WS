#ifndef YOLO3D_MARKERS_NODE_HPP_
#define YOLO3D_MARKERS_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class Yolo3DMarkersNode : public rclcpp::Node {
public:
    // Keep default arguments here to match your .cpp main function
    explicit Yolo3DMarkersNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
};

#endif // YOLO3D_MARKERS_NODE_HPP_