#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>

class CloudCleanerNode : public rclcpp::Node
{
public:
  CloudCleanerNode() : Node("cloud_cleaner_node")
  {
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/depth/pointcloud_filtered", 10);
    
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/depth/pointcloud", 10, std::bind(&CloudCleanerNode::callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "🧹 Universal Cloud Cleaner Started");
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 1. Keep data in Safe Binary Format (PCLPointCloud2)
    // This avoids the 'PointXYZ' vs 'PointXYZRGB' crash entirely.
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
    pcl_conversions::toPCL(*msg, *cloud);

    // 2. Filter using the Binary Format
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2);
    pcl::PassThrough<pcl::PCLPointCloud2> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");     // Filter by Depth
    pass.setFilterLimits(0.1, 10.0);  // Min 0.1m, Max 10m
    pass.filter(*cloud_filtered);

    // 3. Publish
    sensor_msgs::msg::PointCloud2 output;
    pcl_conversions::fromPCL(*cloud_filtered, output);
    output.header = msg->header;
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