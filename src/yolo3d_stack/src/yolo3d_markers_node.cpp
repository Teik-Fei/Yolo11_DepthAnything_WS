/*#include "yolo3d_stack/yolo3d_markers_node.hpp"

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
*/

#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // Crucial for doTransform

using namespace std::chrono_literals;

class Yolo3DMarkersNode : public rclcpp::Node
{
public:
    Yolo3DMarkersNode(const rclcpp::NodeOptions & options)
    : Node("yolo3d_markers_node", options)
    {
        // 1. Initialize TF Buffer and Listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
            "/yolo3d/detections", 10,
            std::bind(&Yolo3DMarkersNode::detectionCallback, this, std::placeholders::_1)
        );

        pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/yolo3d/markers", 10);
        
        RCLCPP_INFO(get_logger(), "🟢 Enhanced YOLO 3D Marker Publisher (Map Frame) Running");
    }

private:
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        visualization_msgs::msg::MarkerArray arr;
        int id_counter = 0;
        
        // 2. Look up transform from Camera -> Map
        geometry_msgs::msg::TransformStamped transform;
        try {
            // Wait up to 0.1s for the transform to be available
            transform = tf_buffer_->lookupTransform(
                "map",                    // Target Frame (Fixed World)
                msg->header.frame_id,     // Source Frame (Camera)
                tf2::TimePointZero        // Get latest available
            );
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(get_logger(), "Could not transform: %s", ex.what());
            return;
        }

        for (auto & det : msg->detections) {
            
            // --- TRANSFORM POSE TO MAP FRAME ---
            geometry_msgs::msg::PoseStamped pose_in, pose_out;
            pose_in.header = msg->header;
            pose_in.pose = det.bbox.center;

            // Perform the math to move point from Camera to Map
            tf2::doTransform(pose_in, pose_out, transform);

            // --- CREATE CUBE MARKER ---
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";     // IMPORTANT: Now defined in Map frame
            m.header.stamp = this->get_clock()->now();
            m.ns = "obstacles";
            m.id = id_counter++;
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.action = visualization_msgs::msg::Marker::ADD;
            
            m.pose = pose_out.pose;        // Use the transformed pose
            
            m.scale.x = det.bbox.size.x;
            m.scale.y = det.bbox.size.y;
            m.scale.z = det.bbox.size.z;
            
            // Color logic: You can customize this based on detection class ID
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.6;
            
            // Lifetime: 0 = Infinite (Stays forever until you restart RViz)
            // Set to 5.0 if you want them to fade after 5 seconds
            m.lifetime = rclcpp::Duration(5.0, 0); 

            arr.markers.push_back(m);

            // --- CREATE TEXT MARKER (Label) ---
            visualization_msgs::msg::Marker text_m = m;
            text_m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_m.id = id_counter++;
            text_m.pose.position.z += (det.bbox.size.z / 2.0) + 0.2; // Float above the box
            if (!det.results.empty()) {
                text_m.text = "Object: " + det.results[0].hypothesis.class_id;
            } else {
                text_m.text = "Unknown Object";
            }
            text_m.scale.z = 0.2; // Text height
            text_m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; // White text
            
            arr.markers.push_back(text_m);
        }

        pub_->publish(arr);
    }

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    
    // Create the default options object
    rclcpp::NodeOptions options;
    
    // Pass 'options' into the constructor
    rclcpp::spin(std::make_shared<Yolo3DMarkersNode>(options));
    
    rclcpp::shutdown();
    return 0;
}
