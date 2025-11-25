#include "yolo3d_stack/yolo11_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <rclcpp/qos.hpp>
#include <fstream>

Yolo11Node::Yolo11Node(const rclcpp::NodeOptions & options) 
    : Node("yolo11_node", options) 
{
    // Reliable QoS for RQT compatibility
    rclcpp::QoS qos_profile(10);

    // Parameters
    std::string cam_topic = this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
    std::string model_path = this->declare_parameter<std::string>("model_path", "");
    conf_thres_ = this->declare_parameter<float>("confidence_threshold", 0.45);
    
    // We ignore the 'use_cuda' param for YOLO to prevent crashing
    // use_cuda_ = this->declare_parameter<bool>("use_cuda", true);

    RCLCPP_INFO(this->get_logger(), "Initializing YOLO Node...");

    // Load Network
    if (!model_path.empty()) {
        try {
            net_ = cv::dnn::readNet(model_path);
            if (net_.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load YOLO model!");
            } else {
                // FORCE CPU BACKEND TO AVOID FUSELAYERS CRASH
                RCLCPP_INFO(this->get_logger(), "Forcing CPU backend for YOLO stability...");
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV Error: %s", e.what());
        }
    }

    // COCO Classes
    classes_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"};

    rmw_qos_profile_t qos = rmw_qos_profile_default;
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        cam_topic, 
        rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos), qos), 
        std::bind(&Yolo11Node::imageCb, this, std::placeholders::_1));
        
    det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/yolo/detections", qos_profile);
    debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/yolo/debug_image", qos_profile);
}

void Yolo11Node::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (net_.empty()) return;

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge fail: %s", e.what());
        return;
    }

    cv::Mat frame = cv_ptr->image;
    if (frame.empty()) return;
    
    // Preprocess
    int col = frame.cols;
    int row = frame.rows;
    int max_len = MAX(col, row);
    cv::Mat resized_img = cv::Mat::zeros(max_len, max_len, CV_8UC3);
    frame.copyTo(resized_img(cv::Rect(0, 0, col, row)));

    cv::Mat blob;
    cv::dnn::blobFromImage(resized_img, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    
    net_.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    try {
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", e.what());
        return;
    }

    // Post-process
    cv::Mat output_data = outputs[0];
    int rows = output_data.size[2]; 
    int dimensions = output_data.size[1];

    if (output_data.dims > 2) {
        output_data = output_data.reshape(0, dimensions);
        output_data = output_data.t();
    }
    
    float* data = (float*)output_data.data;
    float x_factor = (float)max_len / 640.0;
    float y_factor = (float)max_len / 640.0;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float* classes_scores = data + 4;
        cv::Mat scores(1, classes_.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (max_class_score > conf_thres_) {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thres_, 0.45, nms_result);

    vision_msgs::msg::Detection2DArray detections_msg;
    detections_msg.header = msg->header;
    cv::Mat debug_frame = frame.clone();

    for (int idx : nms_result) {
        vision_msgs::msg::Detection2D detection;
        cv::Rect box = boxes[idx];
        
        cv::rectangle(debug_frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = classes_[class_ids[idx]] + ": " + std::to_string(confidences[idx]).substr(0, 4);
        cv::putText(debug_frame, label, cv::Point(box.x, box.y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        detection.bbox.center.position.x = box.x + box.width / 2.0;
        detection.bbox.center.position.y = box.y + box.height / 2.0;
        detection.bbox.size_x = box.width;
        detection.bbox.size_y = box.height;
        
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = classes_[class_ids[idx]];
        hypothesis.hypothesis.score = confidences[idx];
        detection.results.push_back(hypothesis);
        
        detections_msg.detections.push_back(detection);
    }

    if (debug_pub_) {
        sensor_msgs::msg::Image::SharedPtr debug_msg = 
            cv_bridge::CvImage(msg->header, "bgr8", debug_frame).toImageMsg();
        debug_pub_->publish(*debug_msg);
    }
    
    if (det_pub_) {
        det_pub_->publish(detections_msg);
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(Yolo11Node)