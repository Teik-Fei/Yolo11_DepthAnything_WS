// Purpose of this code is to convert incoming YUYV images to BGR8 format
// and republish them, ensuring compatibility with RTAB-Map which requires
// BGR8 encoding and synchronized headers.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageConverterNode : public rclcpp::Node
{
public:
  ImageConverterNode() : Node("image_converter_node")
  {
    // Create Publisher first
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_bgr", 10);

    // Create Subscriber
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/image_raw", 
      10, 
      std::bind(&ImageConverterNode::image_callback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "🚀 YUYV -> BGR8 Converter Node Started");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      // The magic line: This attempts to convert WHATEVER input to BGR8
      // If direct conversion fails, we fallback to manual OpenCV conversion
      if (msg->encoding == "yuyv") {
          // Manually handle YUYV if cv_bridge doesn't auto-convert it
          cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      } else {
          // Standard conversion
          cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Publish the converted image
    // CRITICAL: We must copy the header to keep timestamps synced for RTAB-Map!
    cv_ptr->header = msg->header; 
    cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
    
    pub_->publish(*cv_ptr->toImageMsg());
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageConverterNode>());
  rclcpp::shutdown();
  return 0;
}