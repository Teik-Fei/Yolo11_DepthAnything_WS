// Yolo11 Header file to run on CPU
/*
#ifndef YOLO11_NODE_HPP_
#define YOLO11_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <mutex>
#include <vector>
#include <string>

class Yolo11Node : public rclcpp::Node {
public:
    explicit Yolo11Node(const rclcpp::NodeOptions & options);

private:
    // Callbacks
    void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det3d_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    // Network + Labels
    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    float conf_thres_;

    // Depth buffer
    cv::Mat depth_img_;
    std::mutex depth_mutex_;

    // Camera intrinsic parameters
    float fx_, fy_, cx_, cy_;
};

#endif // YOLO11_NODE_HPP_ */

// Yolo11 Header file to run on TensorRT
#ifndef YOLO11_NODE_HPP_
#define YOLO11_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection3_d.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>
#include <string>
#include <memory>

/// Simple struct for 2D detections coming out of TensorRT
struct YoloDetection2D
{
    int class_id;
    float confidence;
    cv::Rect box;  // in original image coordinates
};

/// Forward declaration of TensorRT wrapper (implemented in .cpp)
class TrtYolo11;

class Yolo11Node : public rclcpp::Node {
public:
    explicit Yolo11Node(const rclcpp::NodeOptions & options);

private:
    // Callbacks
    void rgbCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
    void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

    // Helper to convert 2D+depth into Detection3DArray
    vision_msgs::msg::Detection3DArray
    buildDetections3D(const std_msgs::msg::Header & header,
                      const std::vector<YoloDetection2D> & dets,
                      const cv::Mat & depth);

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det3d_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    // TensorRT YOLO
    std::unique_ptr<TrtYolo11> trt_yolo_;
    std::vector<std::string> classes_;
    int num_classes_{0};

    // Thresholds
    float conf_thres_{0.3f};
    float iou_thres_{0.45f};

    // Network input size
    int input_w_{640};
    int input_h_{480};

    // Depth buffer + mutex
    cv::Mat depth_img_;
    std::mutex depth_mutex_;

    // Camera intrinsics (for depth → 3D)
    float fx_{525.0f};
    float fy_{525.0f};
    float cx_{319.5f};
    float cy_{239.5f};
};

#endif // YOLO11_NODE_HPP_ 
