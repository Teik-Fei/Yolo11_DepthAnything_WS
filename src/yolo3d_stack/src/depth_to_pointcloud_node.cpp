// This Node: Depth + RGB → Colored PointCloud

#include "yolo3d_stack/depth_to_pointcloud_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <limits>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

DepthToPointcloudNode::DepthToPointcloudNode(const rclcpp::NodeOptions & options)
: Node("depth_to_pointcloud_node", options)
{
    // Parameters
    depth_topic_ = declare_parameter("depth_topic","/depth/image_raw");
    rgb_topic_   = declare_parameter("rgb_topic",  "/camera/image_raw");
    cloud_topic_ = declare_parameter("cloud_topic","/depth/pointcloud");

    fx_ = declare_parameter("fx", 555.715735370142);
    fy_ = declare_parameter("fy", 555.6151962989876);
    cx_ = declare_parameter("cx", 346.7216404016699);
    cy_ = declare_parameter("cy", 239.7857718290915);

    step_ = declare_parameter("pixel_step", 3);

    // Publisher
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic_, 10);

    // Subscribers
    depth_sub_.subscribe(this, depth_topic_);
    rgb_sub_.subscribe(this, rgb_topic_);

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image>;

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), depth_sub_, rgb_sub_);

    sync_->registerCallback(
        std::bind(&DepthToPointcloudNode::callback, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "🟦 Depth + RGB → Colored PointCloud node running");
}

void DepthToPointcloudNode::callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg)
{
    // Convert Depth Image
    cv_bridge::CvImageConstPtr cv_depth;
    try {
        cv_depth = cv_bridge::toCvShare(depth_msg, "32FC1");
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth cv_bridge error: %s", e.what());
        return;
    }

    // Convert RGB Image
    cv_bridge::CvImageConstPtr cv_rgb;
    try {
        cv_rgb = cv_bridge::toCvShare(rgb_msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "RGB cv_bridge error: %s", e.what());
        return;
    }

    const cv::Mat &depth = cv_depth->image;
    const cv::Mat &rgb   = cv_rgb->image;

    if (depth.rows != rgb.rows || depth.cols != rgb.cols) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "Depth and RGB resolution mismatch");
        return;
    }

    // Create PointCloud2 (XYZ + RGB)
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = depth_msg->header;
    cloud.is_bigendian = false;
    cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(2, "xyz", "rgb");

    // Compute number of points
    int h = depth.rows;
    int w = depth.cols;

    int height_samples = (h + step_ - 1) / step_;
    int width_samples  = (w + step_ - 1) / step_;
    int num_points = height_samples * width_samples;

    mod.resize(num_points);
    cloud.height = height_samples;
    cloud.width  = width_samples;

    sensor_msgs::PointCloud2Iterator<float> X(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> Y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> Z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> R(cloud, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> G(cloud, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> B(cloud, "b");

    // Fill point cloud
    for (int v = 0; v < h; v += step_) {
        for (int u = 0; u < w; u += step_) {

            float d = depth.at<float>(v, u);

            if (d <= 0.1 || d > 50.0 || std::isnan(d)) {
                *X = std::numeric_limits<float>::quiet_NaN();
                *Y = std::numeric_limits<float>::quiet_NaN();
                *Z = std::numeric_limits<float>::quiet_NaN();
                *R = *G = *B = 0;
            } else {
                // Back-project
                *X = (u - cx_) * d / fx_;
                *Y = (v - cy_) * d / fy_;
                *Z = d;

                // Assign color
                const cv::Vec3b &color = rgb.at<cv::Vec3b>(v, u);
                *R = color[2];
                *G = color[1];
                *B = color[0];
            }

            ++X; ++Y; ++Z;
            ++R; ++G; ++B;
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
