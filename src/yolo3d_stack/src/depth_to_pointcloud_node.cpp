#include "yolo3d_stack/depth_to_pointcloud_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <limits> // Required for NaN

DepthToPointcloudNode::DepthToPointcloudNode(const rclcpp::NodeOptions & options)
: Node("depth_to_pointcloud_node", options)
{
    depth_topic_ = declare_parameter("depth_topic","/depth/image_raw");
    cloud_topic_ = declare_parameter("cloud_topic","/depth/pointcloud");
    fx_ = declare_parameter("fx",600.0);
    fy_ = declare_parameter("fy",600.0);
    cx_ = declare_parameter("cx",320.0);
    cy_ = declare_parameter("cy",240.0);
    step_ = declare_parameter("pixel_step", 10); // Default to 10 for performance

    sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, rclcpp::SensorDataQoS(),
        std::bind(&DepthToPointcloudNode::callback,this,std::placeholders::_1)
    );

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic_,10);

    RCLCPP_INFO(get_logger(),"🟦 Depth → PointCloud node running");
}

void DepthToPointcloudNode::callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    const cv::Mat &depth = cv_ptr->image;

    // Create Cloud
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = msg->header;
    cloud.is_dense = false;
    cloud.is_bigendian = false;

    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");

    // --- CRITICAL: CALCULATE SIZE & RESIZE ---
    int height_samples = (depth.rows + step_ - 1) / step_;
    int width_samples  = (depth.cols + step_ - 1) / step_;
    int num_points = height_samples * width_samples;

    mod.resize(num_points);
    cloud.height = height_samples;
    cloud.width = width_samples;
    // -----------------------------------------

    sensor_msgs::PointCloud2Iterator<float> X(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> Y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> Z(cloud, "z");

    for(int v = 0; v < depth.rows; v += step_) {
        for(int u = 0; u < depth.cols; u += step_) {
            
            float d = depth.at<float>(v, u);

            // Check for invalid depth
            if(d <= 0.1 || d > 20.0 || std::isnan(d)) {
                *X = std::numeric_limits<float>::quiet_NaN();
                *Y = std::numeric_limits<float>::quiet_NaN();
                *Z = std::numeric_limits<float>::quiet_NaN();
            } else {
                *X = (u - cx_) * d / fx_;
                *Y = (v - cy_) * d / fy_;
                *Z = d;
            }

            ++X; ++Y; ++Z;
        }
    }

    pub_->publish(cloud);
}

int main(int argc,char**argv){
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<DepthToPointcloudNode>());
    rclcpp::shutdown();
    return 0;
}