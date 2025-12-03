#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp> // NEW
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex> // NEW for depth sync

// ====================== TensorRT Logger ==========================
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            RCLCPP_INFO(rclcpp::get_logger("TrtLogger"), "[TRT] %s", msg);
        }
    }
};

// ====================== YOLO Detection struct =====================
struct YoloDet {
    int class_id;
    float score;
    float x, y, w, h;
};

// Simple NMS
static void nms(std::vector<YoloDet>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(), [](const YoloDet& a, const YoloDet& b) { return a.score > b.score; });
    std::vector<YoloDet> result;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            // Intersection over Union logic (simplified)
            float xx1 = std::max(dets[i].x - dets[i].w/2, dets[j].x - dets[j].w/2);
            float yy1 = std::max(dets[i].y - dets[i].h/2, dets[j].y - dets[j].h/2);
            float xx2 = std::min(dets[i].x + dets[i].w/2, dets[j].x + dets[j].w/2);
            float yy2 = std::min(dets[i].y + dets[i].h/2, dets[j].y + dets[j].h/2);
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;
            float union_area = (dets[i].w * dets[i].h) + (dets[j].w * dets[j].h) - inter;
            if (inter / union_area > iou_thresh) removed[j] = true;
        }
    }
    dets.swap(result);
}

// ====================== Node Class ================================
class Yolo11TrtNode : public rclcpp::Node {
public:
    Yolo11TrtNode() : Node("yolo11_trt_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing YOLO11 3D TensorRT Node");

        // --- Parameters ---
        engine_path_ = this->declare_parameter<std::string>("engine_path", "");
        conf_thresh_ = this->declare_parameter<float>("conf_threshold", 0.45f);
        visualize_ = this->declare_parameter<bool>("visualize_output", true);
        
        // Intrinsics (Matches params.yaml)
        fx_ = this->declare_parameter<float>("fx", 600.0);
        fy_ = this->declare_parameter<float>("fy", 600.0);
        cx_ = this->declare_parameter<float>("cx", 320.0);
        cy_ = this->declare_parameter<float>("cy", 240.0);

        class_names_ = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        init_trt();

        // --- ROS Interfaces ---
        // 1. RGB Subscriber
        image_sub_ = image_transport::create_subscription(this, "/camera/image_raw",
            std::bind(&Yolo11TrtNode::image_callback, this, std::placeholders::_1), "raw");

        // 2. Depth Subscriber (NEW)
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/depth/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&Yolo11TrtNode::depth_callback, this, std::placeholders::_1));

        // 3. Publishers
        det2d_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/yolo/detections", 10);
        det3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("/yolo3d/detections", 10); // NEW
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/yolo/debug_image", 10);
    }

private:
    // --- Data ---
    cv::Mat latest_depth_;
    std::mutex depth_mutex_;
    float fx_, fy_, cx_, cy_;

    // --- TensorRT variables (Same as before) ---
    TrtLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    int input_w_ = 640, input_h_ = 640; 
    int num_classes_, num_anchors_;
    size_t input_size_, output_size_;
    std::string engine_path_;
    float conf_thresh_;
    bool visualize_;
    std::vector<std::string> class_names_;
    image_transport::Subscriber image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det2d_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det3d_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    // --- Helper: Robust Depth Extraction ---
float get_robust_depth(const cv::Mat& depth_img, cv::Point center, int roi_size = 10) {
    
    // Define the Box
    // Create a square around the center point
    cv::Rect roi_rect(
        center.x - roi_size / 2, 
        center.y - roi_size / 2, 
        roi_size, 
        roi_size
    );

    // Safety Clamp
    // We intersect (&) the requested box with the actual image boundaries.
    // If the box hangs off the edge, this chops off the excess.
    cv::Rect img_bounds(0, 0, depth_img.cols, depth_img.rows);
    cv::Rect final_roi = roi_rect & img_bounds;

    // Sanity check: If box is totally outside, area is 0
    if (final_roi.area() == 0) return -1.0f;

    // Extract the ROI
    cv::Mat roi = depth_img(final_roi);
    
    // Filter Valid Data
    std::vector<float> valid_depths;
    valid_depths.reserve(final_roi.area()); // Optimization: reserve memory

    for (int r = 0; r < roi.rows; ++r) {
        const float* row_ptr = roi.ptr<float>(r);
        for (int c = 0; c < roi.cols; ++c) {
            float d = row_ptr[c];

            // Filter: Must be finite (not NaN), > 0.1m (min range), < 20.0m (max range)
            if (std::isfinite(d) && d > 0.1f && d < 20.0f) {
                valid_depths.push_back(d);
            }
        }
    }

    // Calculate Median
    if (valid_depths.empty()) return -1.0f; // Failure code

    // We don't need to fully sort the vector to find the median.
    // nth_element is faster (O(N)) than sort (O(N log N)).
    size_t n = valid_depths.size() / 2;
    std::nth_element(valid_depths.begin(), valid_depths.begin() + n, valid_depths.end());

    return valid_depths[n];
}

    // --- Init TRT (Simplified for brevity, assumes working engine) ---
    void init_trt() {
        std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
        if(!file.good()) throw std::runtime_error("Engine not found: " + engine_path_);
        size_t size = file.tellg(); file.seekg(0);
        std::vector<char> data(size); file.read(data.data(), size);
        
        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(data.data(), size);
        context_ = engine_->createExecutionContext();
        
        // Setup Buffer Sizes (Assuming standard YOLO export)
        input_size_ = 1 * 3 * input_h_ * input_w_ * sizeof(float);
        // We assume output is [1, 84, 8400]
        num_classes_ = 80; num_anchors_ = 8400; 
        output_size_ = 1 * (num_classes_ + 4) * num_anchors_ * sizeof(float);

        cudaStreamCreate(&stream_);
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);
    }

    // --- CALLBACKS ---

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Store depth for projection
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_ = cv_ptr->image.clone();
        } catch (...) {}
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvShare(msg, "bgr8"); } catch (...) { return; }
        cv::Mat frame = cv_ptr->image;

        // 1. Preprocess
        cv::Mat rsz; 
        cv::resize(frame, rsz, cv::Size(input_w_, input_h_));
        rsz.convertTo(rsz, CV_32F, 1.0/255.0);
        
        std::vector<float> blob(3*input_w_*input_h_);
        for(int c=0; c<3; c++) {
            for(int h=0; h<input_h_; h++) {
                for(int w=0; w<input_w_; w++) {
                    blob[c*input_h_*input_w_ + h*input_w_ + w] = rsz.at<cv::Vec3f>(h,w)[c];
                }
            }
        }

        // 2. Inference
        cudaMemcpyAsync(d_input_, blob.data(), input_size_, cudaMemcpyHostToDevice, stream_);
        
        // Set tensor addresses (TRT 8.5+)
        context_->setInputTensorAddress("images", d_input_);
        context_->setOutputTensorAddress("output0", d_output_);
        context_->enqueueV3(stream_);
        
        std::vector<float> cpu_out(output_size_/sizeof(float));
        cudaMemcpyAsync(cpu_out.data(), d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 3. Decode & Post-Process
        std::vector<YoloDet> detections;
        float scale_x = (float)frame.cols / input_w_;
        float scale_y = (float)frame.rows / input_h_;
        
        int ch = num_classes_ + 4; // 84
        for(int i=0; i<num_anchors_; i++) {
            // Find max class score
            float max_score = 0; int cls_id = -1;
            for(int c=0; c<num_classes_; c++) {
                float s = cpu_out[(4+c)*num_anchors_ + i];
                if(s > max_score) { max_score = s; cls_id = c; }
            }
            if(max_score < conf_thresh_) continue;

            float x = cpu_out[0*num_anchors_ + i] * scale_x;
            float y = cpu_out[1*num_anchors_ + i] * scale_y;
            float w = cpu_out[2*num_anchors_ + i] * scale_x;
            float h = cpu_out[3*num_anchors_ + i] * scale_y;

            detections.push_back({cls_id, max_score, x, y, w, h});
        }
        nms(detections, 0.45);

        // 4. Publish 2D & Calculate 3D
        vision_msgs::msg::Detection2DArray msg_2d; 
        vision_msgs::msg::Detection3DArray msg_3d; // NEW
        msg_2d.header = msg->header;
        msg_3d.header = msg->header;
        
        cv::Mat debug_img = frame.clone();
        cv::Mat depth_copy;
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            if(!latest_depth_.empty()) depth_copy = latest_depth_.clone();
        }

        for(auto& d : detections) {
            // -- 2D Message --
            vision_msgs::msg::Detection2D det2;
            det2.bbox.center.position.x = d.x; det2.bbox.center.position.y = d.y;
            det2.bbox.size_x = d.w; det2.bbox.size_y = d.h;
            msg_2d.detections.push_back(det2);

            // -- 3D Calculation --
            if(!depth_copy.empty()) 
            {
                int cx_px = (int)d.x; 
                int cy_px = (int)d.y;

                // Check bounds
                if(cx_px >= 0 && cx_px < depth_copy.cols && cy_px >= 0 && cy_px < depth_copy.rows) 
                {
                    /*float Z = depth_copy.at<float>(cy_px, cx_px);
        
                    // Filter: Ignore noise (too close) or void (too far)
                    if(Z > 0.1 && Z < 20.0) 
                    { 
                        float X = (cx_px - cx_) * Z / fx_;
                        float Y = (cy_px - cy_) * Z / fy_;

                        vision_msgs::msg::Detection3D det3; */

                    // Calculate Dynamic ROI Size
                    // Use 30% of the smallest side (width or height)
                    int side = std::min(d.w, d.h);
                    int dynamic_roi = (int)(side * 0.3);

                    // Safety Clamp (Minimum Size)
                    // Don't let it get smaller than 1x1 pixel
                    if (dynamic_roi < 1) dynamic_roi = 1;
                    
                    // Get Robust Depth 
                    float Z = get_robust_depth(depth_copy, cv::Point((int)d.x, (int)d.y), dynamic_roi);

                   // Check if valid (The helper returns -1.0 on failure)
                    if (Z > 0.0f) 
                    { 
                        float X = ((int)d.x - cx_) * Z / fx_;
                        float Y = ((int)d.y - cy_) * Z / fy_;

                        vision_msgs::msg::Detection3D det3;
                        det3.header = msg->header;
            
                        // Set 3D Position
                        det3.bbox.center.position.x = X;
                        det3.bbox.center.position.y = Y;
                        det3.bbox.center.position.z = Z;
            
                        // Set Approximate Size (Optional: could assume fixed size based on class)
                        det3.bbox.size.x = 0.2; 
                        det3.bbox.size.y = 0.5;
                        det3.bbox.size.z = 0.2;

                        // CRITICAL: Attach Class ID (The "Name Tag")
                        vision_msgs::msg::ObjectHypothesisWithPose hyp;
            
                        // Safety check for class names
                        if(d.class_id >= 0 && d.class_id < (int)class_names_.size()) {
                            hyp.hypothesis.class_id = class_names_[d.class_id];
                        } else {
                            hyp.hypothesis.class_id = std::to_string(d.class_id);
                        }
                        hyp.hypothesis.score = d.score;
            
                        det3.results.push_back(hyp); // <--- For Fusion node
            
                        msg_3d.detections.push_back(det3); 

                        // Debug text
                        std::string dist = std::to_string(Z).substr(0,3) + "m";
                        cv::putText(debug_img, dist, cv::Point(d.x, d.y), 0, 0.5, {0,0,255}, 2);
                    }
                }
            }

            if(visualize_) {
                // Define the bounding box
                cv::Rect box(d.x - d.w/2, d.y - d.h/2, d.w, d.h);
                cv::rectangle(debug_img, box, {0, 255, 0}, 2);

                // Get the Label (with safety check)
                std::string label;
                if(d.class_id >= 0 && d.class_id < class_names_.size()) {
                    label = class_names_[d.class_id];
                } else {
                    label = "ID:" + std::to_string(d.class_id); // Fallback if name is missing
                }
                
                // Add confidence score (e.g., "person 0.85")
                label += " " + std::to_string(d.score).substr(0,4);

                // Draw Text Background (Green bar so text is readable)
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int top = std::max(box.y, labelSize.height);
                
                cv::rectangle(debug_img, 
                              cv::Point(box.x, top - labelSize.height - 5),
                              cv::Point(box.x + labelSize.width, top),
                              cv::Scalar(0, 255, 0), cv::FILLED);

                // Draw the Text (Black text on Green background)
                cv::putText(debug_img, label, 
                            cv::Point(box.x, top - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        }

        det2d_pub_->publish(msg_2d);
        det3d_pub_->publish(msg_3d); // Publish 3D
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", debug_img).toImageMsg());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Yolo11TrtNode>());
    rclcpp::shutdown();
    return 0;
}
