//=========================================
// YOLO11 Detection Node using TensorRT
//=========================================
/*
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
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <numeric>

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
        conf_thresh_ = this->declare_parameter<float>("conf_threshold", 0.75f);
        visualize_ = this->declare_parameter<bool>("visualize_output", true);

        
        // Intrinsics (Matches params.yaml)
        fx_ = this->declare_parameter<float>("fx", 555.715735370142);
        fy_ = this->declare_parameter<float>("fy", 555.6151962989876);
        cx_ = this->declare_parameter<float>("cx", 346.7216404016699);
        cy_ = this->declare_parameter<float>("cy", 239.7857718290915);

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
        obstacle_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/yolo/obstacle_cloud", 10);
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
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
    std::map<int, std::deque<float>> depth_history_; // Stores last N depth values for each class
    const int smooth_window_ = 3; // Average over 3 frames

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

    float get_smoothed_depth(int class_id, float new_depth) {
        // 1. Init buffer if needed
        if (depth_history_.find(class_id) == depth_history_.end()) {
            depth_history_[class_id] = std::deque<float>();
        }

        // 2. Add new value
        auto& buffer = depth_history_[class_id];
        buffer.push_back(new_depth);

        // 3. Keep size fixed
        if (buffer.size() > smooth_window_) {
            buffer.pop_front();
        }

        // 4. Calculate Average
        float sum = 0;
        for (float d : buffer) sum += d;
        return sum / buffer.size();
    }

    void publish_obstacles_as_cloud(const vision_msgs::msg::Detection3DArray& dets) {
    
    if (dets.detections.empty()) return;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    
    for (const auto& det : dets.detections) {
        float cx = det.bbox.center.position.x;
        // In Odom/Map frame, we usually map Z (forward) to Y (forward) or X (forward) depending on config.
        // For standard ROS "Map" frame: X=Forward, Y=Left, Z=Up.
        // Since YOLO Z is forward, we map YOLO Z -> Cloud X.
        float cy = det.bbox.center.position.z; 
        
        float w = (det.bbox.size.x > 0) ? det.bbox.size.x : 0.3f;
        float d = (det.bbox.size.z > 0) ? det.bbox.size.z : 0.3f;

        int points_per_dim = 5; 
        for (int i = 0; i < points_per_dim; i++) {
            for (int j = 0; j < points_per_dim; j++) {
                pcl::PointXYZ p;
                
                // Map YOLO (Camera Frame) to PointCloud (Base Link Frame approximation)
                // Camera Z (Forward) -> Robot X (Forward)
                // Camera X (Right)   -> Robot Y (Left/Right) - check signs based on your specific TF
                
                // Simple version (Forward = Forward):
                p.x = cy + (d * (j / (float)points_per_dim) - d/2.0f); // Forward
                p.y = -(cx + (w * (i / (float)points_per_dim) - w/2.0f)); // Left/Right (Invert X for standard ROS coordinates)
                p.z = 0.0; // Flat on the floor
                
                cloud.points.push_back(p);
            }
        }
    }

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(cloud, output_msg);
    // CRITICAL: This frame must match the frame you did the math for above.
    // If you swapped X/Z manually above, call this "base_link".
    // If you kept X/Z as camera coordinates, call this "base_link".
    output_msg.header.frame_id = "base_link";
    output_msg.header.stamp = this->now(); // Ideally use msg_3d.header.stamp if available

    obstacle_pub_->publish(output_msg);
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
        
        // For normal yolo bounding box detection
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
        
        // Set tensor addresses
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
                        Z = get_smoothed_depth(d.class_id, Z);
                        
                        // 2. Calculate 3D Position (X, Y)
                        float X = ((int)d.x - cx_) * Z / fx_;
                        float Y = ((int)d.y - cy_) * Z / fy_;

                        vision_msgs::msg::Detection3D det3;
                        det3.header = msg->header;
            
                        // Set 3D Position
                        det3.bbox.center.position.x = X;
                        det3.bbox.center.position.y = Y;
                        det3.bbox.center.position.z = Z;
            
                        // --- NEW: DYNAMIC SIZE CALCULATION ---
                        // Use Similar Triangles to find real size from pixel size
                        
                        // Real Width = (Pixel_Width * Distance) / Focal_Length_X
                        det3.bbox.size.x = (d.w * Z) / fx_; 

                        // Real Height = (Pixel_Height * Distance) / Focal_Length_Y
                        det3.bbox.size.y = (d.h * Z) / fy_;

                        // Estimate Thickness (Z-axis)
                        // Since we can't see "depth" directly, we assume the object 
                        // is roughly as thick as it is wide (Square Footprint).
                        //det3.bbox.size.z = det3.bbox.size.x;
                        det3.bbox.size.z = Z;

                        // Attach Class ID (The "Name Tag")
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
                
                // Add confidence score (e.g "person 0.85")
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
        // Publish the cloud for Nav2
        publish_obstacles_as_cloud(msg_3d); // Assuming you gathered detections into a msg_3d object
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Yolo11TrtNode>());
    rclcpp::shutdown();
    return 0;
}
*/

//=========================================
// YOLO segmentation Node using TensorRT
//=========================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
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
#include <mutex>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <numeric>

// ====================== TensorRT Logger ==========================
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            RCLCPP_INFO(rclcpp::get_logger("TrtLogger"), "[TRT] %s", msg);
        }
    }
};

// ====================== SEGMENTATION CONSTANTS ====================
const int NUM_MASKS = 32;   
const int SEG_H = 160;      
const int SEG_W = 160;      

// ====================== YOLO Detection struct =====================
struct YoloDet {
    int class_id;
    float score;
    float x, y, w, h;
    std::vector<float> mask_coeffs; // Holds the 32 numbers for this object
    cv::Mat mask;                   // The final binary mask (cropped to box)
};

// ====================== NMS FUNCTION ==============================
// (Re-enabled and updated for the new struct)
static void nms(std::vector<YoloDet>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(), [](const YoloDet& a, const YoloDet& b) { return a.score > b.score; });
    std::vector<YoloDet> result;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            // Intersection over Union logic
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
        RCLCPP_INFO(this->get_logger(), "Initializing YOLO11 3D SEGMENTATION Node");

        engine_path_ = this->declare_parameter<std::string>("engine_path", "");
        conf_thresh_ = this->declare_parameter<float>("conf_threshold", 0.75f);
        visualize_ = this->declare_parameter<bool>("visualize_output", true);

        // Intrinsics
        fx_ = this->declare_parameter<float>("fx", 555.715735370142);
        fy_ = this->declare_parameter<float>("fy", 555.6151962989876);
        cx_ = this->declare_parameter<float>("cx", 346.7216404016699);
        cy_ = this->declare_parameter<float>("cy", 239.7857718290915);

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

        image_sub_ = image_transport::create_subscription(this, "/camera/image_raw",
            std::bind(&Yolo11TrtNode::image_callback, this, std::placeholders::_1), "raw");

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/depth/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&Yolo11TrtNode::depth_callback, this, std::placeholders::_1));

        det2d_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/yolo/detections", 10);
        det3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("/yolo3d/detections", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/yolo/debug_image", 10);
        obstacle_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/yolo/obstacle_cloud", 10);
        // Add this line
        visual_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/yolo/masked_cloud", 10);

        last_frame_time_ = this->now();
    }

private:
    cv::Mat latest_depth_;
    std::mutex depth_mutex_;
    float fx_, fy_, cx_, cy_;

    TrtLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    void* d_output1_ = nullptr; // For Masks
    
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
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr visual_cloud_pub_;
    
    std::map<int, std::deque<float>> depth_history_;
    const int smooth_window_ = 3; 
    rclcpp::Time last_frame_time_;
    float current_fps_ = 0.0f;

    // --- Helper: Decode Masks ---
    cv::Mat process_mask(const std::vector<float>& coeffs, const float* proto_masks, cv::Rect box, cv::Size img_size) {
        
        // 1. Matrix Multiplication
        cv::Mat proto(32, 25600, CV_32F, (void*)proto_masks);
        cv::Mat coeff_mat(1, 32, CV_32F, (void*)coeffs.data());
        cv::Mat mask_raw = coeff_mat * proto;

        // 2. Reshape & Sigmoid
        cv::Mat mask_160 = mask_raw.reshape(1, 160);
        cv::exp(-mask_160, mask_160);
        mask_160 = 1.0 / (1.0 + mask_160);

        // 3. Resize to image size
        cv::Mat mask_full;
        cv::resize(mask_160, mask_full, img_size);

        // 4. Crop to Bounding Box
        cv::Mat final_mask = cv::Mat::zeros(img_size, CV_8U);
        cv::Rect roi = box & cv::Rect(0,0, img_size.width, img_size.height);
        
        if (roi.area() > 0) {
            cv::Mat roi_crop = mask_full(roi);
            cv::Mat roi_binary;
            // Threshold probability 0.5
            cv::threshold(roi_crop, roi_binary, 0.2, 255, cv::THRESH_BINARY);
            roi_binary.convertTo(roi_binary, CV_8U);
            roi_binary.copyTo(final_mask(roi));
        }
        return final_mask;
    }

    // --- NEW: Helper for MASKED Depth ---
    // This is better than get_robust_depth because it ignores background!
    float get_masked_depth(const cv::Mat& depth_img, const cv::Mat& mask, cv::Rect box) {
        
        // Safety check
        cv::Rect img_bounds(0, 0, depth_img.cols, depth_img.rows);
        cv::Rect roi = box & img_bounds;
        if (roi.area() == 0) return -1.0f;

        // Extract ROI from Depth and Mask
        cv::Mat depth_roi = depth_img(roi);
        cv::Mat mask_roi = mask(roi);

        std::vector<float> valid_depths;
        valid_depths.reserve(roi.area());

        for (int r = 0; r < depth_roi.rows; ++r) {
            const float* d_ptr = depth_roi.ptr<float>(r);
            const uint8_t* m_ptr = mask_roi.ptr<uint8_t>(r);
            
            for (int c = 0; c < depth_roi.cols; ++c) {
                // KEY CHECK: Is this pixel inside the Segmentation Mask?
                if (m_ptr[c] > 0) { 
                    float d = d_ptr[c];
                    // Standard validity check
                    if (std::isfinite(d) && d > 0.1f && d < 20.0f) {
                        valid_depths.push_back(d);
                    }
                }
            }
        }

        if (valid_depths.empty()) return -1.0f;

        // Median Calculation
        size_t n = valid_depths.size() / 2;
        std::nth_element(valid_depths.begin(), valid_depths.begin() + n, valid_depths.end());
        return valid_depths[n];
    }

    float get_smoothed_depth(int class_id, float new_depth) {
        if (depth_history_.find(class_id) == depth_history_.end()) {
            depth_history_[class_id] = std::deque<float>();
        }
        auto& buffer = depth_history_[class_id];
        buffer.push_back(new_depth);
        if (buffer.size() > smooth_window_) buffer.pop_front();
        float sum = 0;
        for (float d : buffer) sum += d;
        return sum / buffer.size();
    }

    void publish_obstacles_as_cloud(const vision_msgs::msg::Detection3DArray& dets) {
        if (dets.detections.empty()) return;
        pcl::PointCloud<pcl::PointXYZ> cloud;
        for (const auto& det : dets.detections) {
            float cx = det.bbox.center.position.x;
            float cy = det.bbox.center.position.z; 
            float w = (det.bbox.size.x > 0) ? det.bbox.size.x : 0.3f;
            float d = (det.bbox.size.z > 0) ? det.bbox.size.z : 0.3f;
            int points_per_dim = 5; 
            for (int i = 0; i < points_per_dim; i++) {
                for (int j = 0; j < points_per_dim; j++) {
                    pcl::PointXYZ p;
                    p.x = cy + (d * (j / (float)points_per_dim) - d/2.0f); 
                    p.y = -(cx + (w * (i / (float)points_per_dim) - w/2.0f)); 
                    p.z = 0.5; 
                    cloud.points.push_back(p);
                }
            }
        }
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(cloud, output_msg);
        output_msg.header.frame_id = "base_link";
        output_msg.header.stamp = this->now(); 
        obstacle_pub_->publish(output_msg);
    }

    // --- NEW: Generate Clean 3D Object Cloud ---
    void publish_masked_visual_cloud(const std::vector<YoloDet>& detections, const cv::Mat& depth_img, std_msgs::msg::Header header) {
        
        if (detections.empty() || depth_img.empty()) return;

        // Create PCL Cloud (XYZ + RGB for color)
        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        for (const auto& det : detections) {
            // Safety: Ensure box is within image bounds
            cv::Rect box(det.x - det.w/2, det.y - det.h/2, det.w, det.h);
            box = box & cv::Rect(0, 0, depth_img.cols, depth_img.rows);
            if (box.area() <= 0) continue;

            // Get ROIs
            cv::Mat depth_roi = depth_img(box);
            cv::Mat mask_roi = det.mask(box); // This is the Binary Mask (0 or 255)

            // Iterate over every pixel in the box
            for (int r = 0; r < depth_roi.rows; r += 2) { // Step 2 for speed (downsample)
                const float* d_ptr = depth_roi.ptr<float>(r);
                const uint8_t* m_ptr = mask_roi.ptr<uint8_t>(r);

                for (int c = 0; c < depth_roi.cols; c += 2) {
                    
                    // CRITICAL FILTER: Only project if Mask says "Object" (value > 0)
                    if (m_ptr[c] > 0) {
                        float Z = d_ptr[c];

                        // Sanity Check depth
                        if (std::isfinite(Z) && Z > 0.1 && Z < 10.0) {
                            
                            // 2D pixel -> 3D point (Pinhole Model)
                            // We need absolute image coordinates (roi_x + c)
                            float u = box.x + c;
                            float v = box.y + r;

                            pcl::PointXYZRGB p;
                            p.z = Z;
                            p.x = (u - cx_) * Z / fx_;
                            p.y = (v - cy_) * Z / fy_;

                            switch(det.class_id % 3) {
                                case 0: // ID 0, 3, 6... (Person?) -> RED
                                    p.r = 255; p.g = 0; p.b = 0; 
                                    break;
                                case 1: // ID 1, 4, 7... (Bicycle?) -> GREEN
                                    p.r = 0; p.g = 255; p.b = 0;
                                    break;
                                case 2: // ID 2, 5, 8... (Car?) -> BLUE
                                    p.r = 0; p.g = 100; p.b = 255;
                                    break;
                            }
                            // Add intensity variation so it looks 3D
                            p.r = std::max(0, p.r - (int)(Z * 20));

                            cloud.points.push_back(p);
                        }
                    }
                }
            }
        }

        // Convert to ROS Msg
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(cloud, output_msg);
        output_msg.header = header;
        output_msg.header.frame_id = "camera_link_optical"; // Points are relative to camera
        
        visual_cloud_pub_->publish(output_msg);
    }

    void init_trt() {
        std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
        if(!file.good()) throw std::runtime_error("Engine not found: " + engine_path_);
        size_t size = file.tellg(); file.seekg(0);
        std::vector<char> data(size); file.read(data.data(), size);
        
        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(data.data(), size);
        context_ = engine_->createExecutionContext();
        
        // 1. Input
        input_size_ = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    
        // 2. Output 0 (Boxes + Class + Mask Coeffs)
        int output0_rows = 4 + 80 + 32; 
        num_anchors_ = 8400;
        num_classes_ = 80;
        output_size_ = 1 * output0_rows * num_anchors_ * sizeof(float);

        // 3. Output 1 (Proto-Masks)
        size_t output1_size = 1 * 32 * 160 * 160 * sizeof(float);

        cudaStreamCreate(&stream_);
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);
        cudaMalloc(&d_output1_, output1_size);
    }

    // --- CALLBACKS ---
    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_ = cv_ptr->image.clone();
        } catch (...) {}
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        rclcpp::Time now = this->now();
    
        // Safety Check: Ensure clock types match before math
        if (last_frame_time_.get_clock_type() != now.get_clock_type()) {
            last_frame_time_ = now; // Reset if types mismatch
            return; // Skip FPS for first frame
        }

        double dt = (now - last_frame_time_).seconds();
    
        if (dt > 0.0) {
            float instantaneous_fps = 1.0f / dt;
            if (current_fps_ == 0.0f) current_fps_ = instantaneous_fps;
            else current_fps_ = 0.9f * current_fps_ + 0.1f * instantaneous_fps;
        }
        last_frame_time_ = now;

        cv_bridge::CvImageConstPtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvShare(msg, "bgr8"); } catch (...) { return; }
        cv::Mat frame = cv_ptr->image;

        // Init Messages (Fixed: These were missing!)
        vision_msgs::msg::Detection2DArray msg_2d; 
        vision_msgs::msg::Detection3DArray msg_3d; 
        msg_2d.header = msg->header;
        msg_3d.header = msg->header;

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
        context_->setInputTensorAddress("images", d_input_);
        context_->setOutputTensorAddress("output0", d_output_);
        context_->setOutputTensorAddress("output1", d_output1_);
        context_->enqueueV3(stream_);
        
        std::vector<float> cpu_out0(output_size_/sizeof(float));
        cudaMemcpyAsync(cpu_out0.data(), d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
        
        std::vector<float> cpu_out1(32 * 160 * 160);
        cudaMemcpyAsync(cpu_out1.data(), d_output1_, cpu_out1.size()*sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 3. Parsing
        std::vector<YoloDet> detections;
        float scale_x = (float)frame.cols / input_w_;
        float scale_y = (float)frame.rows / input_h_;
        
        for(int i=0; i<num_anchors_; i++) {
            float max_score = 0; int cls_id = -1;
            for(int c=0; c<num_classes_; c++) {
                float s = cpu_out0[(4+c)*num_anchors_ + i];
                if(s > max_score) { max_score = s; cls_id = c; }
            }
            if(max_score < conf_thresh_) continue;

            float x = cpu_out0[0*num_anchors_ + i] * scale_x;
            float y = cpu_out0[1*num_anchors_ + i] * scale_y;
            float w = cpu_out0[2*num_anchors_ + i] * scale_x;
            float h = cpu_out0[3*num_anchors_ + i] * scale_y;

            std::vector<float> coeffs(32);
            for(int k=0; k<32; k++) {
                coeffs[k] = cpu_out0[(84 + k)*num_anchors_ + i];
            }
            detections.push_back({cls_id, max_score, x, y, w, h, coeffs});
        }
        nms(detections, 0.45);

        // 4. Post-Process & 3D Projection
        cv::Mat debug_img = frame.clone();
        cv::Mat depth_copy;
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            if(!latest_depth_.empty()) depth_copy = latest_depth_.clone();
        }

        for (auto& det : detections) {
            cv::Rect box(det.x - det.w/2, det.y - det.h/2, det.w, det.h);
            
            // A. Generate Mask
            det.mask = process_mask(det.mask_coeffs, cpu_out1.data(), box, frame.size());
            
            // B. Populate 2D Msg
            vision_msgs::msg::Detection2D det2;
            det2.bbox.center.position.x = det.x; 
            det2.bbox.center.position.y = det.y;
            det2.bbox.size_x = det.w; 
            det2.bbox.size_y = det.h;
            msg_2d.detections.push_back(det2);

            // C. Calculate 3D (Using MASK for accuracy!)
            if(!depth_copy.empty()) {
                // Use get_masked_depth instead of get_robust_depth
                float Z = get_masked_depth(depth_copy, det.mask, box);

                if(Z > 0.0f) {
                    Z = get_smoothed_depth(det.class_id, Z);
                    float X = ((int)det.x - cx_) * Z / fx_;
                    float Y = ((int)det.y - cy_) * Z / fy_;

                    vision_msgs::msg::Detection3D det3;
                    det3.header = msg->header;
                    det3.bbox.center.position.x = X;
                    det3.bbox.center.position.y = Y;
                    det3.bbox.center.position.z = Z;
                    det3.bbox.size.x = (det.w * Z) / fx_;
                    det3.bbox.size.y = (det.h * Z) / fy_;
                    det3.bbox.size.z = Z;

                    vision_msgs::msg::ObjectHypothesisWithPose hyp;
                    if(det.class_id >= 0 && det.class_id < (int)class_names_.size()) {
                        hyp.hypothesis.class_id = class_names_[det.class_id];
                    } else {
                        hyp.hypothesis.class_id = std::to_string(det.class_id);
                    }
                    hyp.hypothesis.score = det.score;
                    det3.results.push_back(hyp);
                    msg_3d.detections.push_back(det3);

                    // Debug text
                    std::string dist = std::to_string(Z).substr(0,3) + "m";
                    cv::putText(debug_img, dist, cv::Point(det.x, det.y), 0, 0.5, {0,0,255}, 2);
                }
            }

            // D. Visualize
            if(visualize_) {
                // 1. CLAMP THE BOX
                // Intersect with image bounds to ensure it's valid
                cv::Rect safe_box = box & cv::Rect(0, 0, debug_img.cols, debug_img.rows);

                if (safe_box.area() > 0) {
                    // Draw Box
                    cv::rectangle(debug_img, safe_box, {0, 255, 0}, 2);
                    
                    // Draw Mask Overlay (Use safe_box, NOT box)
                    cv::Mat roi_mask = det.mask(safe_box); // Safe crop
                    
                    cv::Mat color_mask;
                    cv::cvtColor(roi_mask, color_mask, cv::COLOR_GRAY2BGR);
                    
                    // Color the mask green
                    color_mask.setTo(cv::Scalar(0, 255, 0), roi_mask > 0);
                    
                    // Blend only the safe region
                    cv::addWeighted(debug_img(safe_box), 1.0, color_mask, 0.5, 0.0, debug_img(safe_box));

                    // Draw Label
                    std::string label = (det.class_id < class_names_.size()) ? class_names_[det.class_id] : "ID";
                    cv::putText(debug_img, label, cv::Point(safe_box.x, safe_box.y-5), 0, 0.5, {0,0,0}, 1);
                }
            }
        }

        if(visualize_) {
            std::string fps_text = "FPS: " + std::to_string((int)current_fps_);
        
            // Shadow
            cv::putText(debug_img, fps_text, cv::Point(12, 32), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 4);
            // Text
            cv::putText(debug_img, fps_text, cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        det2d_pub_->publish(msg_2d);
        // Publish the CLEAN visual cloud (Replacing the V-Shape)
        if (!depth_copy.empty()) {
            publish_masked_visual_cloud(detections, depth_copy, msg->header);
        }
        det3d_pub_->publish(msg_3d);
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", debug_img).toImageMsg());
        publish_obstacles_as_cloud(msg_3d);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        // Try to start the node
        auto node = std::make_shared<Yolo11TrtNode>();
        rclcpp::spin(node);
    } 
    catch (const std::exception& e) {
        // If it crashes, print WHY
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "NODE CRASHED: %s", e.what());
    }
    catch (...) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "NODE CRASHED: Unknown exception");
    }

    rclcpp::shutdown();
    return 0;
}