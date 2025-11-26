/*#include "yolo3d_stack/yolo11_node.hpp"
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
} */

// ^ Original Code CAA 261125

#include "yolo3d_stack/yolo11_node.hpp"

#include "rclcpp_components/register_node_macro.hpp"
#include <rclcpp/qos.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <numeric>
#include <algorithm>

// =========================
//   TensorRT Logger
// =========================

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            RCLCPP_INFO(rclcpp::get_logger("TrtLogger"), "[TRT] %s", msg);
        }
    }
};

// =========================
//   TensorRT YOLO wrapper
// =========================

class TrtYolo11
{
public:
    TrtYolo11(const std::string & engine_path,
              int input_w,
              int input_h,
              float conf_thres,
              float iou_thres)
        : input_w_(input_w),
          input_h_(input_h),
          conf_thres_(conf_thres),
          iou_thres_(iou_thres)
    {
        // Load engine file into memory
        std::ifstream file(engine_path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open TensorRT engine: " + engine_path);

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_)
            throw std::runtime_error("Failed to create TensorRT runtime");

        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size, nullptr);
        if (!engine_)
            throw std::runtime_error("Failed to deserialize engine");

        context_ = engine_->createExecutionContext();
        if (!context_)
            throw std::runtime_error("Failed to create execution context");

        // Assume: 1 input (0), 1 output (1)
        input_index_  = engine_->getBindingIndex("images"); // may differ
        if (input_index_ == -1)
            input_index_ = 0;

        output_index_ = 1 - input_index_;

        auto input_dims = engine_->getBindingDimensions(input_index_);
        auto output_dims = engine_->getBindingDimensions(output_index_);

        // Compute sizes
        input_size_ = 1;
        for (int i = 0; i < input_dims.nbDims; ++i)
            input_size_ *= input_dims.d[i];       // [1,3,H,W] → 1*3*H*W

        output_size_ = 1;
        for (int i = 0; i < output_dims.nbDims; ++i)
            output_size_ *= output_dims.d[i];     // [1, C, N] or [1,N,C]

        // Infer layout assuming [1, C, N] as in your original OpenCV code:
        //   C = 4 + num_classes, N = #anchors (e.g. 8400)
        if (output_dims.nbDims == 3)
        {
            out_c_ = output_dims.d[1];
            out_n_ = output_dims.d[2];
        }
        else
        {
            // Fallback, user can print these to verify
            out_c_ = -1;
            out_n_ = -1;
        }

        // Allocate device buffers
        cudaMalloc(&device_buffers_[input_index_],  input_size_  * sizeof(float));
        cudaMalloc(&device_buffers_[output_index_], output_size_ * sizeof(float));
        cudaStreamCreate(&stream_);

        host_output_.resize(output_size_);
    }

    ~TrtYolo11()
    {
        if (device_buffers_[input_index_])  cudaFree(device_buffers_[input_index_]);
        if (device_buffers_[output_index_]) cudaFree(device_buffers_[output_index_]);
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) context_->destroy();
        if (engine_)  engine_->destroy();
        if (runtime_) runtime_->destroy();
    }

    int outC() const { return out_c_; }
    int outN() const { return out_n_; }

    /// Run inference on a BGR image. Returns detections in original-image coordinates.
    void infer(const cv::Mat & bgr,
               std::vector<YoloDetection2D> & out_dets,
               int num_classes,
               const std::vector<std::string> & class_names)
    {
        out_dets.clear();
        if (bgr.empty())
            return;

        // 1) Preprocess (resize + normalize to [0,1], NCHW float)
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(input_w_, input_h_));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        std::vector<float> host_input(input_size_);
        // Layout: [1,3,H,W]
        int channels = 3;
        int h = input_h_;
        int w = input_w_;

        for (int c = 0; c < channels; ++c)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float val = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;
                    host_input[c * h * w + y * w + x] = val;
                }
            }
        }

        // 2) Copy to device
        cudaMemcpyAsync(device_buffers_[input_index_],
                        host_input.data(),
                        input_size_ * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_);

        // 3) Run inference
        context_->enqueueV2(device_buffers_, stream_, nullptr);

        // 4) Copy output back
        cudaMemcpyAsync(host_output_.data(),
                        device_buffers_[output_index_],
                        output_size_ * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);

        // 5) Decode (assuming [1, C, N] with C = 4 + num_classes)
        if (out_c_ <= 0 || out_n_ <= 0)
        {
            RCLCPP_WARN(rclcpp::get_logger("TrtYolo11"),
                        "Unexpected output dims, cannot decode");
            return;
        }

        const float * data = host_output_.data();
        int bbox_dim = 4;
        int cls_dim  = out_c_ - bbox_dim;

        float x_scale = static_cast<float>(bgr.cols) / static_cast<float>(input_w_);
        float y_scale = static_cast<float>(bgr.rows) / static_cast<float>(input_h_);

        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> scores;

        for (int i = 0; i < out_n_; ++i)
        {
            const float * row = data + i * out_c_;
            float cx = row[0];
            float cy = row[1];
            float w_box = row[2];
            float h_box = row[3];

            // Find best class
            float best_score = 0.0f;
            int best_id = -1;

            for (int c = 0; c < cls_dim; ++c)
            {
                float score = row[bbox_dim + c];
                if (score > best_score)
                {
                    best_score = score;
                    best_id = c;
                }
            }

            if (best_id < 0 || best_score < conf_thres_)
                continue;

            // Convert from cx, cy, w, h to x,y,w,h
            float x = cx - w_box * 0.5f;
            float y = cy - h_box * 0.5f;

            // Scale back to original image
            int x0 = static_cast<int>(x * x_scale);
            int y0 = static_cast<int>(y * y_scale);
            int ww = static_cast<int>(w_box * x_scale);
            int hh = static_cast<int>(h_box * y_scale);

            // Clamp
            x0 = std::max(0, std::min(x0, bgr.cols - 1));
            y0 = std::max(0, std::min(y0, bgr.rows - 1));
            if (x0 + ww > bgr.cols) ww = bgr.cols - x0;
            if (y0 + hh > bgr.rows) hh = bgr.rows - y0;

            boxes.emplace_back(x0, y0, ww, hh);
            class_ids.push_back(best_id);
            scores.push_back(best_score);
        }

        // 6) NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres_, iou_thres_, nms_indices);

        for (int idx : nms_indices)
        {
            YoloDetection2D det;
            det.class_id = class_ids[idx];
            det.confidence = scores[idx];
            det.box = boxes[idx];
            out_dets.push_back(det);
        }
    }

private:
    TrtLogger logger_;
    nvinfer1::IRuntime * runtime_{nullptr};
    nvinfer1::ICudaEngine * engine_{nullptr};
    nvinfer1::IExecutionContext * context_{nullptr};

    void * device_buffers_[2]{nullptr, nullptr};
    cudaStream_t stream_{nullptr};

    size_t input_size_{0};
    size_t output_size_{0};

    int input_w_{0};
    int input_h_{0};

    int out_c_{-1};  // C = 4 + num_classes
    int out_n_{-1};  // N = #anchors

    float conf_thres_;
    float iou_thres_;

    std::vector<float> host_output_;
}; 

/*
// =========================
//   TensorRT YOLO wrapper (TensorRT 10.x Compatible)
// =========================

class TrtYolo11
{
public:
    TrtYolo11(const std::string & engine_path,
              int input_w,
              int input_h,
              float conf_thres,
              float iou_thres)
        : input_w_(input_w),
          input_h_(input_h),
          conf_thres_(conf_thres),
          iou_thres_(iou_thres)
    {
        // Load engine file into memory
        std::ifstream file(engine_path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open TensorRT engine: " + engine_path);

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        // TensorRT 10.x: Create runtime
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_)
            throw std::runtime_error("Failed to create TensorRT runtime");

        // TensorRT 10.x: Deserialize engine
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_)
            throw std::runtime_error("Failed to deserialize engine");

        // TensorRT 10.x: Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_)
            throw std::runtime_error("Failed to create execution context");

        // Get binding information
        num_bindings_ = engine_->getNbBindings();
        
        RCLCPP_INFO(rclcpp::get_logger("TrtYolo11"), "Number of bindings: %d", num_bindings_);

        // Find input and output bindings
        for (int i = 0; i < num_bindings_; ++i) {
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            const char* name = engine_->getBindingName(i);
            bool is_input = engine_->bindingIsInput(i);
            
            if (is_input) {
                input_index_ = i;
                RCLCPP_INFO(rclcpp::get_logger("TrtYolo11"), "Found input: %s at index %d", name, i);
            } else {
                output_index_ = i;
                RCLCPP_INFO(rclcpp::get_logger("TrtYolo11"), "Found output: %s at index %d", name, i);
            }
        }

        // Get dimensions
        auto input_dims = engine_->getBindingDimensions(input_index_);
        auto output_dims = engine_->getBindingDimensions(output_index_);

        // Compute sizes
        input_size_ = 1;
        for (int i = 0; i < input_dims.nbDims; ++i) {
            input_size_ *= input_dims.d[i];
        }

        output_size_ = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size_ *= output_dims.d[i];
        }

        // Infer output layout - handle different output formats
        if (output_dims.nbDims == 3) {
            // Format: [batch, features, num_boxes]
            out_c_ = output_dims.d[1];  // Number of features per box (4 + num_classes)
            out_n_ = output_dims.d[2];  // Number of boxes
        } else if (output_dims.nbDims == 2) {
            // Format: [num_boxes, features]
            out_n_ = output_dims.d[0];
            out_c_ = output_dims.d[1];
        } else {
            RCLCPP_WARN(rclcpp::get_logger("TrtYolo11"), 
                       "Unexpected output dimensions: %d", output_dims.nbDims);
            out_c_ = -1;
            out_n_ = -1;
        }

        RCLCPP_INFO(rclcpp::get_logger("TrtYolo11"), 
                   "Input size: %zu, Output size: %zu, Output dims: C=%d, N=%d", 
                   input_size_, output_size_, out_c_, out_n_);

        // Allocate device buffers
        cudaMalloc(&device_buffers_[input_index_], input_size_ * sizeof(float));
        cudaMalloc(&device_buffers_[output_index_], output_size_ * sizeof(float));
        cudaStreamCreate(&stream_);

        host_output_.resize(output_size_);
    }

    ~TrtYolo11()
    {
        // Cleanup CUDA resources
        if (device_buffers_[input_index_]) cudaFree(device_buffers_[input_index_]);
        if (device_buffers_[output_index_]) cudaFree(device_buffers_[output_index_]);
        if (stream_) cudaStreamDestroy(stream_);
        
        // TensorRT 10.x: Objects are automatically managed
    }

    int outC() const { return out_c_; }
    int outN() const { return out_n_; }

    /// Run inference on a BGR image
    void infer(const cv::Mat & bgr,
               std::vector<YoloDetection2D> & out_dets,
               int num_classes,
               const std::vector<std::string> & class_names)
    {
        out_dets.clear();
        if (bgr.empty())
            return;

        // 1) Preprocess
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(input_w_, input_h_));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        std::vector<float> host_input(input_size_);
        int channels = 3;
        int h = input_h_;
        int w = input_w_;

        // Fill input tensor (NCHW format)
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float val = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;
                    host_input[c * h * w + y * w + x] = val;
                }
            }
        }

        // 2) Copy to device
        cudaMemcpyAsync(device_buffers_[input_index_],
                        host_input.data(),
                        input_size_ * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_);

        // 3) Run inference - TensorRT 10.x
        void* bindings[] = { 
            device_buffers_[input_index_], 
            device_buffers_[output_index_] 
        };
        
        bool success = context_->enqueueV2(bindings, stream_, nullptr);
        
        if (!success) {
            RCLCPP_ERROR(rclcpp::get_logger("TrtYolo11"), "TensorRT inference execution failed");
            return;
        }

        // 4) Copy output back
        cudaMemcpyAsync(host_output_.data(),
                        device_buffers_[output_index_],
                        output_size_ * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);

        // 5) Decode output based on detected format
        if (out_c_ <= 0 || out_n_ <= 0) {
            RCLCPP_WARN(rclcpp::get_logger("TrtYolo11"), "Invalid output dimensions, skipping decode");
            return;
        }

        const float * data = host_output_.data();
        int bbox_dim = 4;  // cx, cy, w, h
        int cls_dim = out_c_ - bbox_dim;

        if (cls_dim <= 0) {
            RCLCPP_WARN(rclcpp::get_logger("TrtYolo11"), "Invalid class dimension: %d", cls_dim);
            return;
        }

        float x_scale = static_cast<float>(bgr.cols) / static_cast<float>(input_w_);
        float y_scale = static_cast<float>(bgr.rows) / static_cast<float>(input_h_);

        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> scores;

        // Process each detection
        for (int i = 0; i < out_n_; ++i) {
            const float * row = data + i * out_c_;
            
            float cx = row[0];
            float cy = row[1];
            float w_box = row[2];
            float h_box = row[3];

            // Find best class
            float best_score = 0.0f;
            int best_id = -1;

            for (int c = 0; c < cls_dim; ++c) {
                float score = row[bbox_dim + c];
                if (score > best_score) {
                    best_score = score;
                    best_id = c;
                }
            }

            if (best_id < 0 || best_score < conf_thres_)
                continue;

            // Convert from center format to corner format
            float x = cx - w_box * 0.5f;
            float y = cy - h_box * 0.5f;

            // Scale back to original image coordinates
            int x0 = static_cast<int>(x * x_scale);
            int y0 = static_cast<int>(y * y_scale);
            int ww = static_cast<int>(w_box * x_scale);
            int hh = static_cast<int>(h_box * y_scale);

            // Clamp coordinates to image boundaries
            x0 = std::max(0, std::min(x0, bgr.cols - 1));
            y0 = std::max(0, std::min(y0, bgr.rows - 1));
            ww = std::max(1, std::min(ww, bgr.cols - x0));
            hh = std::max(1, std::min(hh, bgr.rows - y0));

            boxes.emplace_back(x0, y0, ww, hh);
            class_ids.push_back(best_id);
            scores.push_back(best_score);
        }

        // 6) Apply Non-Maximum Suppression
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres_, iou_thres_, nms_indices);

        // 7) Create final detections
        for (int idx : nms_indices) {
            YoloDetection2D det;
            det.class_id = class_ids[idx];
            det.confidence = scores[idx];
            det.box = boxes[idx];
            out_dets.push_back(det);
        }

        RCLCPP_DEBUG(rclcpp::get_logger("TrtYolo11"), 
                    "Detection complete: %zu detections after NMS", out_dets.size());
    }

private:
    TrtLogger logger_;
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    void* device_buffers_[2]{nullptr, nullptr};
    cudaStream_t stream_{nullptr};

    size_t input_size_{0};
    size_t output_size_{0};
    int num_bindings_{0};

    int input_w_{0};
    int input_h_{0};
    int input_index_{0};
    int output_index_{1};

    int out_c_{-1};  // Number of features per detection
    int out_n_{-1};  // Number of detections

    float conf_thres_;
    float iou_thres_;

    std::vector<float> host_output_;
}; */

// ===========================================
//     Yolo11Node implementation
// ===========================================

Yolo11Node::Yolo11Node(const rclcpp::NodeOptions & options)
    : Node("yolo11_node", options)
{
    rclcpp::QoS qos_sensor = rclcpp::SensorDataQoS();

    // Parameters
    std::string rgb_topic   = declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
    std::string depth_topic = declare_parameter<std::string>("depth_topic", "/depth/image_raw");
    std::string det_topic   = declare_parameter<std::string>("detections_3d_topic", "/yolo/detections3d");
    std::string dbg_topic   = declare_parameter<std::string>("debug_image_topic", "/yolo/debug_image");

    std::string engine_path = declare_parameter<std::string>("engine_path", "yolo3d_stack/models/yolo11_fp16.engine");
    std::string class_path  = declare_parameter<std::string>("class_names_path", "yolo3d_stack/models/coco.names");

    conf_thres_ = declare_parameter<float>("conf_thres", 0.3f);
    iou_thres_  = declare_parameter<float>("iou_thres",  0.45f);

    input_w_    = declare_parameter<int>("input_width",  640);
    input_h_    = declare_parameter<int>("input_height", 480);

    fx_         = declare_parameter<float>("fx", 525.0f);
    fy_         = declare_parameter<float>("fy", 525.0f);
    cx_         = declare_parameter<float>("cx", 319.5f);
    cy_         = declare_parameter<float>("cy", 239.5f);

    // Load class names
    {
        std::ifstream infile(class_path);
        if (!infile)
        {
            RCLCPP_WARN(get_logger(), "Could not open class names file: %s", class_path.c_str());
        }
        else
        {
            std::string line;
            while (std::getline(infile, line))
            {
                if (!line.empty())
                    classes_.push_back(line);
            }
            num_classes_ = static_cast<int>(classes_.size());
            RCLCPP_INFO(get_logger(), "Loaded %d class names", num_classes_);
        }
    }

    // Init TensorRT YOLO
    try
    {
        trt_yolo_ = std::make_unique<TrtYolo11>(
            engine_path,
            input_w_,
            input_h_,
            conf_thres_,
            iou_thres_);
        RCLCPP_INFO(get_logger(), "TensorRT YOLO11 initialized (engine: %s)", engine_path.c_str());
    }
    catch (const std::exception & e)
    {
        RCLCPP_FATAL(get_logger(), "Failed to init TrtYolo11: %s", e.what());
        throw;
    }

    // Subscriptions
    rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
        rgb_topic, qos_sensor,
        std::bind(&Yolo11Node::rgbCallback, this, std::placeholders::_1));

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic, qos_sensor,
        std::bind(&Yolo11Node::depthCallback, this, std::placeholders::_1));

    // Publishers
    det3d_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(det_topic, 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>(dbg_topic, 10);

    RCLCPP_INFO(get_logger(), "Yolo11Node initialized. Subscribing to:\n  RGB: %s\n  Depth: %s",
                rgb_topic.c_str(), depth_topic.c_str());
}

void Yolo11Node::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
    std::lock_guard<std::mutex> lock(depth_mutex_);
    try
    {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
        if (msg->encoding == "32FC1" || msg->encoding == "TYPE_32FC1")
        {
            depth_img_ = cv_ptr->image.clone();
        }
        else if (msg->encoding == "16UC1")
        {
            // Convert mm → m
            cv::Mat depth_float;
            cv_ptr->image.convertTo(depth_float, CV_32FC1, 1.0 / 1000.0);
            depth_img_ = depth_float;
        }
        else
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Unsupported depth encoding: %s", msg->encoding.c_str());
        }
    }
    catch (const cv_bridge::Exception & e)
    {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception in depthCallback: %s", e.what());
    }
}

void Yolo11Node::rgbCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    }
    catch (const cv_bridge::Exception & e)
    {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception in rgbCallback: %s", e.what());
        return;
    }

    cv::Mat bgr = cv_ptr->image;

    // Run YOLO-TensorRT
    std::vector<YoloDetection2D> dets;
    trt_yolo_->infer(bgr, dets, num_classes_, classes_);

    // Get a local copy of depth
    cv::Mat depth_copy;
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (!depth_img_.empty())
            depth_copy = depth_img_.clone();
    }

    // Build Detection3DArray (if we have depth)
    if (!depth_copy.empty())
    {
        auto arr = buildDetections3D(msg->header, dets, depth_copy);
        det3d_pub_->publish(arr);
    }

    // Draw 2D boxes for debug image
    for (const auto & d : dets)
    {
        cv::rectangle(bgr, d.box, cv::Scalar(0, 255, 0), 2);
        std::string label;
        if (d.class_id >= 0 && d.class_id < (int)classes_.size())
            label = classes_[d.class_id];
        else
            label = "id=" + std::to_string(d.class_id);

        char txt[256];
        std::snprintf(txt, sizeof(txt), "%s %.2f", label.c_str(), d.confidence);
        cv::putText(bgr, txt, cv::Point(d.box.x, std::max(0, d.box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    auto dbg_msg = cv_bridge::CvImage(msg->header, "bgr8", bgr).toImageMsg();
    debug_pub_->publish(*dbg_msg);
}

vision_msgs::msg::Detection3DArray
Yolo11Node::buildDetections3D(const std_msgs::msg::Header & header,
                              const std::vector<YoloDetection2D> & dets,
                              const cv::Mat & depth)
{
    vision_msgs::msg::Detection3DArray arr;
    arr.header = header;

    for (const auto & det : dets)
    {
        int cx_px = det.box.x + det.box.width / 2;
        int cy_px = det.box.y + det.box.height / 2;

        if (cx_px < 0 || cx_px >= depth.cols ||
            cy_px < 0 || cy_px >= depth.rows)
            continue;

        float Z = depth.at<float>(cy_px, cx_px); // meters
        if (Z <= 0.1f || !std::isfinite(Z))
            continue;

        float X = (static_cast<float>(cx_px) - cx_) / fx_ * Z;
        float Y = (static_cast<float>(cy_px) - cy_) / fy_ * Z;

        vision_msgs::msg::Detection3D det3d;
        det3d.header = header;

        // Result
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = std::to_string(det.class_id);
        hyp.hypothesis.score    = det.confidence;
        hyp.pose.pose.position.x = X;
        hyp.pose.pose.position.y = Y;
        hyp.pose.pose.position.z = Z;

        det3d.results.push_back(hyp);

        // Bounding box (rough size from 2D + depth; you can refine)
        det3d.bbox.center.position = hyp.pose.pose.position;
        det3d.bbox.size.x = det.box.width  * Z / fx_;
        det3d.bbox.size.y = det.box.height * Z / fy_;
        det3d.bbox.size.z = 0.5f; // arbitrary depth; tune for your robot

        arr.detections.push_back(det3d);
    }

    return arr;
}

// Register as a component (optional)
RCLCPP_COMPONENTS_REGISTER_NODE(Yolo11Node)

// Standalone executable entry point
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Yolo11Node>(rclcpp::NodeOptions());
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
