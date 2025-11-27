/*#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
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

// ====================== TensorRT Logger ==========================
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Filter out info-level messages
        if (severity <= Severity::kINFO)
        {
            RCLCPP_INFO(rclcpp::get_logger("TrtLogger"), "[TRT] %s", msg);
        }
    }
};

// ====================== YOLO Detection struct =====================
struct YoloDet
{
    int class_id;
    float score;
    float x;      // center x
    float y;      // center y
    float w;      // width
    float h;      // height
};

// Simple NMS
static void nms(std::vector<YoloDet>& dets, float iou_thresh)
{
    auto iou = [](const YoloDet& a, const YoloDet& b) {
        float x1_min = a.x - a.w * 0.5f;
        float y1_min = a.y - a.h * 0.5f;
        float x1_max = a.x + a.w * 0.5f;
        float y1_max = a.y + a.h * 0.5f;

        float x2_min = b.x - b.w * 0.5f;
        float y2_min = b.y - b.h * 0.5f;
        float x2_max = b.x + b.w * 0.5f;
        float y2_max = b.y + b.h * 0.5f;

        float inter_xmin = std::max(x1_min, x2_min);
        float inter_ymin = std::max(y1_min, y2_min);
        float inter_xmax = std::min(x1_max, x2_max);
        float inter_ymax = std::min(y1_max, y2_max);

        float inter_w = std::max(0.0f, inter_xmax - inter_xmin);
        float inter_h = std::max(0.0f, inter_ymax - inter_ymin);
        float inter_area = inter_w * inter_h;

        float area1 = a.w * a.h;
        float area2 = b.w * b.h;

        float union_area = area1 + area2 - inter_area;
        if (union_area <= 0.0f) return 0.0f;
        return inter_area / union_area;
    };

    std::sort(dets.begin(), dets.end(),
              [](const YoloDet& a, const YoloDet& b) { return a.score > b.score; });

    std::vector<YoloDet> result;
    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i)
    {
        if (removed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
        {
            if (removed[j]) continue;
            if (iou(dets[i], dets[j]) > iou_thresh)
            {
                removed[j] = true;
            }
        }
    }

    dets.swap(result);
}

// ====================== Node Class ================================
class Yolo11TrtNode : public rclcpp::Node
{
public:
    Yolo11TrtNode()
        : Node("yolo11_trt_node"),
          logger_(),
          runtime_(nullptr),
          engine_(nullptr),
          context_(nullptr),
          d_input_(nullptr),
          d_output_(nullptr),
          input_size_bytes_(0),
          output_size_bytes_(0),
          input_w_(640),
          input_h_(640)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing YOLO11 TensorRT node (TRT 10 API).");

        // Parameters
        engine_path_ = this->declare_parameter<std::string>(
            "engine_path",
            "/home/mecatron/Yolo11_DepthAnything_WS/src/yolo3d_stack/models/yolo11n_fp16.engine");
        conf_thresh_ = this->declare_parameter<float>("conf_threshold", 0.25f);
        iou_thresh_  = this->declare_parameter<float>("iou_threshold", 0.45f);
        image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");

        visualize_output_ = this->declare_parameter<bool>("visualize_output", true);

        // (Optional) COCO80 class names
        class_names_ = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };

        // Init TensorRT
        init_trt();

        // ROS interfaces
        image_sub_ = image_transport::create_subscription(
            this,
            image_topic_,
            std::bind(&Yolo11TrtNode::image_callback, this, std::placeholders::_1),
            "raw");

        detections_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "/yolo/detections", 10);

        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/yolo/debug_image", 10);
    }

    ~Yolo11TrtNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down YOLO11 TensorRT node.");

        if (d_input_)
        {
            cudaFree(d_input_);
            d_input_ = nullptr;
        }
        if (d_output_)
        {
            cudaFree(d_output_);
            d_output_ = nullptr;
        }

        if (context_)
        {
            delete context_;
            context_ = nullptr;
        }
        if (engine_)
        {
            delete engine_;
            engine_ = nullptr;
        }
        if (runtime_)
        {
            delete runtime_;
            runtime_ = nullptr;
        }

        cudaStreamDestroy(stream_);
    }

private:
    // ---------------- TensorRT Init ----------------
    void init_trt()
    {
        RCLCPP_INFO(this->get_logger(), "Loading TensorRT engine: %s", engine_path_.c_str());

        // Read engine file into memory
        std::ifstream file(engine_path_, std::ios::binary);
        if (!file.good())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open engine file: %s", engine_path_.c_str());
            throw std::runtime_error("Engine file not found");
        }

        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        // Create runtime and engine using TRT 10 API
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_)
        {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_)
        {
            throw std::runtime_error("Failed to deserialize CUDA engine");
        }

        context_ = engine_->createExecutionContext();
        if (!context_)
        {
            throw std::runtime_error("Failed to create execution context");
        }

        // We assume:
        //  input tensor name: "images"
        //  output tensor name: "output0"
        const char* input_name  = "images";
        const char* output_name = "output0";

        if (engine_->getTensorIOMode(input_name) != nvinfer1::TensorIOMode::kINPUT)
        {
            RCLCPP_ERROR(this->get_logger(), "Tensor '%s' is not marked as INPUT in engine.", input_name);
            throw std::runtime_error("Engine IO mismatch");
        }

        if (engine_->getTensorIOMode(output_name) != nvinfer1::TensorIOMode::kOUTPUT)
        {
            RCLCPP_ERROR(this->get_logger(), "Tensor '%s' is not marked as OUTPUT in engine.", output_name);
            throw std::runtime_error("Engine IO mismatch");
        }

        nvinfer1::Dims input_dims  = engine_->getTensorShape(input_name);
        nvinfer1::Dims output_dims = engine_->getTensorShape(output_name);

        // Expect 1x3x640x640
        if (input_dims.nbDims != 4)
        {
            RCLCPP_ERROR(this->get_logger(), "Unexpected input dims nbDims=%d", input_dims.nbDims);
        }
        int in_n = input_dims.d[0];
        int in_c = input_dims.d[1];
        int in_h = input_dims.d[2];
        int in_w = input_dims.d[3];

        RCLCPP_INFO(this->get_logger(),
            "Engine input '%s': NCHW = %d x %d x %d x %d",
            input_name, in_n, in_c, in_h, in_w);

        input_h_ = in_h;
        input_w_ = in_w;

        int64_t input_elems = static_cast<int64_t>(in_n) * in_c * in_h * in_w;
        input_size_bytes_   = input_elems * sizeof(float);

        // Output 1 x 84 x 8400 typically
        if (output_dims.nbDims != 3)
        {
            RCLCPP_ERROR(this->get_logger(), "Unexpected output dims nbDims=%d", output_dims.nbDims);
        }
        int out_n = output_dims.d[0];
        int out_c = output_dims.d[1];
        int out_k = output_dims.d[2];

        RCLCPP_INFO(this->get_logger(),
            "Engine output '%s': NCK = %d x %d x %d",
            output_name, out_n, out_c, out_k);

        num_classes_    = out_c - 4;  // assume: 4 box + num_classes
        num_anchors_    = out_k;
        output_size_bytes_ = static_cast<int64_t>(out_n) * out_c * out_k * sizeof(float);

        RCLCPP_INFO(this->get_logger(),
            "YOLO-style decode: anchors=%d, classes=%d",
            num_anchors_, num_classes_);

        // Allocate GPU buffers
        cudaStreamCreate(&stream_);

        if (cudaMalloc(&d_input_, input_size_bytes_) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed for input");
        }
        if (cudaMalloc(&d_output_, output_size_bytes_) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed for output");
        }
    }

    // ------------- Preprocess: BGR Image → NCHW float ----------------
    void preprocess(const cv::Mat& frame, std::vector<float>& blob, int& resized_w, int& resized_h)
    {
        resized_w = input_w_;
        resized_h = input_h_;

        cv::Mat img;
        cv::resize(frame, img, cv::Size(input_w_, input_h_));

        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // BGR -> RGB if needed (Ultralytics assumes RGB). Here we assume BGR okay.
        // But to be closer to Ultralytics: convert:
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // HWC -> CHW
        blob.resize(3 * input_h_ * input_w_);
        int idx = 0;
        for (int c = 0; c < 3; ++c)
        {
            for (int y = 0; y < input_h_; ++y)
            {
                for (int x = 0; x < input_w_; ++x)
                {
                    blob[idx++] = img.at<cv::Vec3f>(y, x)[c];
                }
            }
        }
    }

    // ------------- Decode output tensor ----------------
    void decode(const float* out, int orig_w, int orig_h, std::vector<YoloDet>& dets)
    {
        dets.clear();

        // Standard YOLO export shape: [1, 84, 8400]
        // 84 = 4 box coords + 80 classes
        // 8400 = num_anchors
    
        // TRT output is flattened. We access it as [Channel][Anchor] usually, 
        // but checks are needed. Based on your code, accessing out[channel * K + i] implies
        // Planar layout. 

        int C = num_classes_ + 4; // 84
        int K = num_anchors_;     // 8400

        // Scale factors to resize the bounding box from 640x640 back to original image
        float scale_x = static_cast<float>(orig_w) / static_cast<float>(input_w_);
        float scale_y = static_cast<float>(orig_h) / static_cast<float>(input_h_);

        for (int i = 0; i < K; ++i)
        {
            // 1. Get Class Score first to save processing time
            int best_class = -1;
            float best_score = 0.0f;

            // Iterate over classes (starting at index 4)
            for (int c = 0; c < num_classes_; ++c)
            {
                // Note: Standard export usually gives probabilities (0-1), not logits.
                // So we usually DO NOT need sigmoid here.
                float score = out[(4 + c) * K + i];
            
                if (score > best_score)
                {
                    best_score = score;
                    best_class = c;
                }
            }

            // Filter by confidence
            if (best_score < conf_thresh_) continue;

            // 2. Get Box Coordinates
            // Standard export: 0=x_center, 1=y_center, 2=width, 3=height (Absolute Pixels)
            float x = out[0 * K + i];
            float y = out[1 * K + i];
            float w = out[2 * K + i];
            float h = out[3 * K + i];

            // 3. Scale to original resolution
            float r_x = x * scale_x;
            float r_y = y * scale_y;
            float r_w = w * scale_x;
            float r_h = h * scale_y;

            YoloDet det;
            det.class_id = best_class;
            det.score    = best_score;
            det.x        = r_x;
            det.y        = r_y;
            det.w        = r_w;
            det.h        = r_h;

            dets.push_back(det);
        }

        if (!dets.empty())
        {
            nms(dets, iou_thresh_);
        }
    }
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        if (!context_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "TensorRT context not initialized.");
            return;
        }

        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        const cv::Mat& frame = cv_ptr->image;
        int orig_w = frame.cols;
        int orig_h = frame.rows;

        // Preprocess
        std::vector<float> input_blob;
        int resized_w, resized_h;
        preprocess(frame, input_blob, resized_w, resized_h);

        cudaMemcpyAsync(d_input_, input_blob.data(), input_size_bytes_,
                    cudaMemcpyHostToDevice, stream_);

        const char* input_name  = "images";
        const char* output_name = "output0";

        if (!context_->setInputTensorAddress(input_name, d_input_) ||
            !context_->setOutputTensorAddress(output_name, d_output_))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set input/output tensor address.");
            return;
        }

        if (!context_->enqueueV3(stream_))
        {
            RCLCPP_ERROR(this->get_logger(), "enqueueV3 failed.");
            return;
        }
        std::vector<float> output_host(output_size_bytes_ / sizeof(float));
        cudaMemcpyAsync(output_host.data(), d_output_, output_size_bytes_,
                    cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Decode
        std::vector<YoloDet> detections;
        decode(output_host.data(), orig_w, orig_h, detections); 

        // Print the first anchor's box coordinates (indices 0-3) and the first class score (index 4)
        // at the center of the image (approx anchor index 4200 for 8400 total)
        int mid_idx = num_anchors_ / 2; 
        float val_x = output_host[0 * num_anchors_ + mid_idx];
        float val_y = output_host[1 * num_anchors_ + mid_idx];
        float val_w = output_host[2 * num_anchors_ + mid_idx];
        float val_conf = output_host[4 * num_anchors_ + mid_idx];

        RCLCPP_INFO(this->get_logger(), 
        "Raw Center Anchor -> X: %.2f Y: %.2f W: %.2f Conf: %.5f", 
        val_x, val_y, val_w, val_conf);

        // ───────────── Publish Detection2DArray ─────────────
        vision_msgs::msg::Detection2DArray det_array;
        det_array.header = msg->header;

        // We'll draw on a copy
        cv::Mat vis;
        if (visualize_output_)
            vis = frame.clone();

        for (const auto& det : detections)
        {
            vision_msgs::msg::Detection2D det_msg;

            det_msg.bbox.center.position.x = det.x;
            det_msg.bbox.center.position.y = det.y;
            det_msg.bbox.size_x = det.w;
            det_msg.bbox.size_y = det.h;

            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = std::to_string(det.class_id);
            hyp.hypothesis.score    = det.score;
            det_msg.results.push_back(hyp);

            det_array.detections.push_back(det_msg);

            // ───── draw on image if enabled ─────
            if (visualize_output_ && !vis.empty())
            {
                float x1 = det.x - det.w * 0.5f;
                float y1 = det.y - det.h * 0.5f;
                float x2 = det.x + det.w * 0.5f;
                float y2 = det.y + det.h * 0.5f;

                cv::rectangle(vis, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2),
                          cv::Scalar(0, 255, 0), 2);

                std::string label;
                if (det.class_id >= 0 && det.class_id < (int)class_names_.size())
                    label = class_names_[det.class_id];
                else
                    label = std::to_string(det.class_id);

                char buf[64];
                std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), det.score);
                cv::putText(vis, buf, cv::Point((int)x1, (int)std::max(y1 - 5.0f, 0.0f)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }

        detections_pub_->publish(det_array);

        // ───────────── Publish debug image ─────────────
        if (visualize_output_ && debug_image_pub_ && !vis.empty())
        {
            cv_bridge::CvImage out_msg;
            out_msg.header   = msg->header;
            out_msg.encoding = "bgr8";
            out_msg.image    = vis;
            debug_image_pub_->publish(*out_msg.toImageMsg());
        }
    }


private:
    // TensorRT
    TrtLogger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;

    void* d_input_;
    void* d_output_;
    size_t input_size_bytes_;
    size_t output_size_bytes_;
    int input_w_;
    int input_h_;
    int num_classes_;
    int num_anchors_;

    // Config
    std::string engine_path_;
    std::string image_topic_;
    float conf_thresh_;
    float iou_thresh_;
    std::vector<std::string> class_names_;

    // ROS
    image_transport::Subscriber image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;

    bool visualize_output_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
};

// ======================= main =====================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Yolo11TrtNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} */

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

            /*// -- 3D Calculation --
            if(!depth_copy.empty()) {
                int cx_px = (int)d.x; 
                int cy_px = (int)d.y;

                if(cx_px >= 0 && cx_px < depth_copy.cols && cy_px >= 0 && cy_px < depth_copy.rows) {
                    float Z = depth_copy.at<float>(cy_px, cx_px);
                    
                    if(Z > 0.1 && Z < 20.0) { // Valid depth
                        float X = (cx_px - cx_) * Z / fx_;
                        float Y = (cy_px - cy_) * Z / fy_;

                        vision_msgs::msg::Detection3D det3;
                        det3.header = msg->header;
                        det3.bbox.center.position.x = X;
                        det3.bbox.center.position.y = Y;
                        det3.bbox.center.position.z = Z;
                        det3.bbox.size.x = 0.2; // Approx size
                        det3.bbox.size.y = 0.5;
                        det3.bbox.size.z = 0.2;

                        // --- NEW: ADD CLASS INFO HERE ---
                        vision_msgs::msg::ObjectHypothesisWithPose hyp;
                        if(d.class_id >= 0 && d.class_id < class_names_.size()) {
                            hyp.hypothesis.class_id = class_names_[d.class_id];
                        } else {
                            hyp.hypothesis.class_id = std::to_string(d.class_id);
                        }
                        hyp.hypothesis.score = d.score;
                        det3.results.push_back(hyp);
                        // --------------------------------
                        
                        msg_3d.detections.push_back(det3); // Add to 3D array

                        // Debug text
                        std::string dist = std::to_string(Z).substr(0,3) + "m";
                        cv::putText(debug_img, dist, cv::Point(d.x, d.y), 0, 0.5, {0,0,255}, 2);
                    }
                }
            } */

            // -- 3D Calculation --
            if(!depth_copy.empty()) 
            {
                int cx_px = (int)d.x; 
                int cy_px = (int)d.y;

                // Check bounds
                if(cx_px >= 0 && cx_px < depth_copy.cols && cy_px >= 0 && cy_px < depth_copy.rows) 
                {
                    float Z = depth_copy.at<float>(cy_px, cx_px);
        
                    // Filter: Ignore noise (too close) or void (too far)
                    if(Z > 0.1 && Z < 20.0) 
                    { 
                        float X = (cx_px - cx_) * Z / fx_;
                        float Y = (cy_px - cy_) * Z / fy_;

                        vision_msgs::msg::Detection3D det3;
                        det3.header = msg->header;
            
                        // 1. Set 3D Position
                        det3.bbox.center.position.x = X;
                        det3.bbox.center.position.y = Y;
                        det3.bbox.center.position.z = Z;
            
                        // 2. Set Approximate Size (Optional: could assume fixed size based on class)
                        det3.bbox.size.x = 0.2; 
                        det3.bbox.size.y = 0.5;
                        det3.bbox.size.z = 0.2;

                        // 3. CRITICAL: Attach Class ID (The "Name Tag")
                        vision_msgs::msg::ObjectHypothesisWithPose hyp;
            
                        // Safety check for class names
                        if(d.class_id >= 0 && d.class_id < (int)class_names_.size()) {
                            hyp.hypothesis.class_id = class_names_[d.class_id];
                        } else {
                            hyp.hypothesis.class_id = std::to_string(d.class_id);
                        }
                        hyp.hypothesis.score = d.score;
            
                        det3.results.push_back(hyp); // <--- Fusion node needs this!
            
                        msg_3d.detections.push_back(det3); 

                        // Debug text
                        std::string dist = std::to_string(Z).substr(0,3) + "m";
                        cv::putText(debug_img, dist, cv::Point(d.x, d.y), 0, 0.5, {0,0,255}, 2);
                    }
                }
            }

            if(visualize_) {
                // 1. Define the bounding box
                cv::Rect box(d.x - d.w/2, d.y - d.h/2, d.w, d.h);
                cv::rectangle(debug_img, box, {0, 255, 0}, 2);

                // 2. Get the Label (with safety check)
                std::string label;
                if(d.class_id >= 0 && d.class_id < class_names_.size()) {
                    label = class_names_[d.class_id];
                } else {
                    label = "ID:" + std::to_string(d.class_id); // Fallback if name is missing
                }
                
                // Optional: Add confidence score (e.g., "person 0.85")
                label += " " + std::to_string(d.score).substr(0,4);

                // 3. Draw Text Background (Green bar so text is readable)
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int top = std::max(box.y, labelSize.height);
                
                cv::rectangle(debug_img, 
                              cv::Point(box.x, top - labelSize.height - 5),
                              cv::Point(box.x + labelSize.width, top),
                              cv::Scalar(0, 255, 0), cv::FILLED);

                // 4. Draw the Text (Black text on Green background)
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
