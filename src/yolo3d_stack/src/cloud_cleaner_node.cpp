// the purpose of this code is to subscribe to a PointCloud2 topic,
// remove NaN points and points beyond a certain distance, and republish the cleaned cloud.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

class CloudCleanerNode : public rclcpp::Node
{
public:
  CloudCleanerNode() : Node("cloud_cleaner_node")
  {
    // 1. Publisher for the CLEAN data
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/depth/pointcloud_filtered", 10);
    
    // 2. Subscriber for the MESSY data
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/depth/pointcloud", 10, std::bind(&CloudCleanerNode::callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "🧹 Cloud Cleaner Started: Removing NaNs and Far points...");
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Convert ROS -> PCL
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

    // FILTER STEP: Remove NaNs and points further than 5 meters
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(temp_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 20.0); // Min 0.1m, Max 20.0m
    pass.filter(*temp_cloud);

    // Convert PCL -> ROS and Publish
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*temp_cloud, output);
    output.header = msg->header; // CRITICAL: Keep the original timestamp!
    pub_->publish(output);
  }
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudCleanerNode>());
  rclcpp::shutdown();
  return 0;
}