#include "yolo3d_stack/yolo3d_markers_node.hpp"

Yolo3DMarkersNode::Yolo3DMarkersNode(const rclcpp::NodeOptions & options)
: Node("yolo3d_markers_node", options)
{
    sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
        "/yolo3d/detections", 10,
        std::bind(&Yolo3DMarkersNode::detectionCallback, this, std::placeholders::_1)
    );

    pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/yolo3d/markers",10);

    RCLCPP_INFO(get_logger(),"🟢 YOLO 3D Marker Publisher Running");
}

void Yolo3DMarkersNode::detectionCallback(
    const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray arr;
    int id = 0;

    for (auto & det : msg->detections) {
        visualization_msgs::msg::Marker m;
        m.header = msg->header;
        m.id = id++;
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.pose.position = det.bbox.center.position;
        m.scale.x = det.bbox.size.x;
        m.scale.y = det.bbox.size.y;
        m.scale.z = det.bbox.size.z;
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.5;

        arr.markers.push_back(m);
    }

    pub_->publish(arr);
}

int main(int argc,char**argv){
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<Yolo3DMarkersNode>());
    rclcpp::shutdown();
    return 0;
}
