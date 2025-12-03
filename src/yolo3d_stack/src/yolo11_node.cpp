#include "yolo3d_stack/yolo11_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <rclcpp/qos.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <mutex>

// ===========================================
//     YOLO2D → YOLO3D (Depth + RGB Fusion)
// ===========================================

Yolo11Node::Yolo11Node(const rclcpp::NodeOptions & options) 
    : Node("yolo11_node", options) 
{
    rclcpp::QoS qos_profile(10);

    // ---------------- Parameters ----------------
    std::string cam_topic   = declare_parameter("camera_topic", "/camera/image_raw");
    std::string depth_topic = declare_parameter("depth_topic", "/depth/image_raw");
    std::string model_path  = declare_parameter("model_path", "/home/mecatron/yolo11n.onnx");
    conf_thres_ = declare_parameter("confidence_threshold", 0.45);

    // Camera intrinsics
    fx_ = declare_parameter("fx", 600.0);
    fy_ = declare_parameter("fy", 600.0);
    cx_ = declare_parameter("cx", 320.0);
    cy_ = declare_parameter("cy", 240.0);

    RCLCPP_INFO(get_logger(), "YOLO3D Node Initializing...");

    // ---------------- Load Model ----------------
    if (!model_path.empty()) {
        try {
            net_ = cv::dnn::readNet(model_path);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            RCLCPP_INFO(get_logger(), "Loaded YOLO model: %s", model_path.c_str());
        } catch (...) {
            RCLCPP_ERROR(get_logger(), "Model Load Failed. Check path: %s", model_path.c_str());
        }
    } else RCLCPP_WARN(get_logger(), "⚠ No model path provided!");

    // ---------------- Class Labels (COCO 80) ----------------
    // Note: Used only for naming. Detection logic uses model output dimensions.
    classes_ = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
        "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
        "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };

    // ---------------- Subscriptions ----------------
    // Use SensorDataQoS for images (best effort/high throughput)
    sub_img_ = create_subscription<sensor_msgs::msg::Image>(
        cam_topic, rclcpp::SensorDataQoS(), 
        std::bind(&Yolo11Node::imageCb, this, std::placeholders::_1));

    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic, rclcpp::SensorDataQoS(),
        std::bind(&Yolo11Node::depthCb, this, std::placeholders::_1));

    // ---------------- Publishers ----------------
    det3d_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("/yolo3d/detections", qos_profile);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/yolo/debug_image", qos_profile);

    RCLCPP_INFO(get_logger(), "YOLO3D READY ✔");
}


// ---------------- Store Latest Depth ----------------
void Yolo11Node::depthCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    try {
        // Ensure we copy the image so it doesn't get overwritten by the transport layer
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg,"32FC1");
        std::lock_guard<std::mutex> lock(depth_mutex_);
        depth_img_ = cv_ptr->image.clone();
    } catch(...) {}
}


// =====================================================
//                  YOLO + DEPTH = 3D
// =====================================================
void Yolo11Node::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) 
{
    if (net_.empty()) return;

    cv_bridge::CvImageConstPtr cv_ptr;
    try { cv_ptr = cv_bridge::toCvShare(msg,"bgr8"); }
    catch(...) { return; }

    cv::Mat frame = cv_ptr->image;
    
    // Create Blob
    cv::Mat blob; 
    cv::dnn::blobFromImage(frame, blob, 1/255.0, {640,640}, cv::Scalar(), true, false);
    net_.setInput(blob);

    // Inference
    std::vector<cv::Mat> out;
    net_.forward(out, net_.getUnconnectedOutLayersNames());
    
    // YOLOv8/11 Output: [1, 4 + classes, 8400]
    cv::Mat out_m = out[0];
    int rows = out_m.size[2]; 
    int dims = out_m.size[1]; // e.g., 84 for COCO

    // Transpose to [8400, 84] for easier iteration
    out_m = out_m.reshape(0, dims);
    out_m = out_m.t();
    
    float* data = (float*)out_m.data;

    std::vector<cv::Rect> boxes; 
    std::vector<int> ids; 
    std::vector<float> confs;

    // Safety: Ensure model output matches logic
    int num_classes = dims - 4; 
    if (num_classes < 1) return;

    for(int i=0; i<rows; i++) {
        // Scores start at index 4
        cv::Mat scores(1, num_classes, CV_32FC1, data + 4);
        cv::Point classIdPoint; 
        double score;
        minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

        if(score > conf_thres_){
            float cx_box = data[0];
            float cy_box = data[1];
            float w = data[2];
            float h = data[3];

            // Convert center-xywh to top-left-xywh
            boxes.emplace_back(cx_box - w * 0.5, cy_box - h * 0.5, w, h);
            ids.push_back(classIdPoint.x); 
            confs.push_back(score);
        }
        data += dims; // Move to next row
    }

    // NMS
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confs, conf_thres_, 0.45, keep);

    // Fetch Depth (Thread Safe)
    cv::Mat depth;
    { 
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if(depth_img_.empty()) return;
        depth = depth_img_.clone();
    }

    // Prepare Messages
    vision_msgs::msg::Detection3DArray arr;
    arr.header = msg->header;
    cv::Mat dbg = frame.clone();

    for(int i : keep){
        cv::Rect b = boxes[i];
        
        // Sampling point: Center of the bounding box
        int center_x = b.x + b.width / 2;
        int center_y = b.y + b.height / 2;

        // Bounds check on Depth Map
        if(center_x < 0 || center_y < 0 || center_x >= depth.cols || center_y >= depth.rows) continue;

        // Get Depth
        float Z = depth.at<float>(center_y, center_x); 
        if(Z < 0.1 || Z > 20.0 || std::isnan(Z)) continue;

        // Back-project 2D -> 3D
        float X = (center_x - cx_) * Z / fx_;
        float Y = (center_y - cy_) * Z / fy_;

        // Create 3D Detection
        vision_msgs::msg::Detection3D d;
        d.bbox.center.position.x = X; 
        d.bbox.center.position.y = Y; 
        d.bbox.center.position.z = Z;
        
        // Approximate physical size
        d.bbox.size.x = b.width * Z / fx_; 
        d.bbox.size.y = b.height * Z / fy_; 
        d.bbox.size.z = d.bbox.size.y * 0.5; // Guess depth thickness

        vision_msgs::msg::ObjectHypothesisWithPose h;
        // Safety check for label name
        if (ids[i] < (int)classes_.size()) {
            h.hypothesis.class_id = classes_[ids[i]];
        } else {
            h.hypothesis.class_id = "unknown_" + std::to_string(ids[i]);
        }
        h.hypothesis.score = confs[i];
        d.results = {h};
        arr.detections.push_back(d);

        // Debug drawing
        cv::rectangle(dbg, b, {0, 255, 0}, 2);
        std::string label_text = h.hypothesis.class_id + " " + std::to_string(Z).substr(0,4) + "m";
        cv::putText(dbg, label_text, {b.x, b.y - 5}, 0, 0.6, {0, 255, 0}, 2);
    }

    det3d_pub_->publish(arr);
    debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", dbg).toImageMsg());
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Yolo11Node>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
} 

