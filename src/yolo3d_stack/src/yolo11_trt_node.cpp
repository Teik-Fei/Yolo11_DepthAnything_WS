#include <rclcpp/rclcpp.hpp>
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

    // ------------- Decode YOLO-style output --------------------------
    void decode(const float* out, int orig_w, int orig_h, std::vector<YoloDet>& dets)
    {
        dets.clear();
        // layout: [1, C, K] where C = 4 + num_classes, K = num_anchors
        // For each anchor i:
        //   x = out[0*K + i]
        //   y = out[1*K + i]
        //   w = out[2*K + i]
        //   h = out[3*K + i]
        //   class scores = out[(4 + c)*K + i]

        int C = num_classes_ + 4;
        int K = num_anchors_;

        for (int i = 0; i < K; ++i)
        {
            float x = out[0 * K + i];
            float y = out[1 * K + i];
            float w = out[2 * K + i];
            float h = out[3 * K + i];

            // Find best class
            int best_class = -1;
            float best_score = 0.0f;

            for (int c = 0; c < num_classes_; ++c)
            {
                float score = out[(4 + c) * K + i];
                if (score > best_score)
                {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_score < conf_thresh_) continue;

            // Assume coords are normalized [0,1] in network input space
            float cx = x * static_cast<float>(orig_w);
            float cy = y * static_cast<float>(orig_h);
            float bw = w * static_cast<float>(orig_w);
            float bh = h * static_cast<float>(orig_h);

            YoloDet det;
            det.class_id = best_class;
            det.score    = best_score;
            det.x        = cx;
            det.y        = cy;
            det.w        = bw;
            det.h        = bh;
            dets.push_back(det);
        }

        if (!dets.empty())
        {
            nms(dets, iou_thresh_);
        }
    }

    // ------------- Image callback -----------------------------------
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

        // Copy to GPU
        cudaMemcpyAsync(d_input_, input_blob.data(), input_size_bytes_,
                        cudaMemcpyHostToDevice, stream_);

        // Bind buffers using TRT 10 API
        const char* input_name  = "images";
        const char* output_name = "output0";

        if (!context_->setInputTensorAddress(input_name, d_input_))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set input tensor address.");
            return;
        }
        if (!context_->setOutputTensorAddress(output_name, d_output_))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set output tensor address.");
            return;
        }

        // Inference
        if (!context_->enqueueV3(stream_))
        {
            RCLCPP_ERROR(this->get_logger(), "enqueueV3 failed.");
            return;
        }

        // Copy output back
        std::vector<float> output_host(output_size_bytes_ / sizeof(float));
        cudaMemcpyAsync(output_host.data(), d_output_, output_size_bytes_,
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Decode
        std::vector<YoloDet> detections;
        decode(output_host.data(), orig_w, orig_h, detections);

        // Publish Detection2DArray
        vision_msgs::msg::Detection2DArray det_array;
        det_array.header = msg->header;

        for (const auto& det : detections)
        {
            /*
            vision_msgs::msg::Detection2D det_msg;
            det_msg.header = msg->header;

            // center + size
            det_msg.bbox.center.x = det.x;
            det_msg.bbox.center.y = det.y;
            det_msg.bbox.size_x   = det.w;
            det_msg.bbox.size_y   = det.h; 

            vision_msgs::msg::Detection2D det_msg;
            det_msg.header = msg->header;

            det_msg.results.resize(1);
            det_msg.results[0].hypothesis.class_id = std::to_string(det.class_id);
            det_msg.results[0].hypothesis.score = det.confidence; 

            // BOUNDING BOX (HUMBLE FORMAT)
            det_msg.bbox.center.position.x = det.x;
            det_msg.bbox.center.position.y = det.y;
            det_msg.bbox.size_x = det.w;
            det_msg.bbox.size_y = det.h;


            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = (det.class_id >= 0 &&
                                       det.class_id < static_cast<int>(class_names_.size()))
                                          ? class_names_[det.class_id]
                                          : std::to_string(det.class_id);
            hyp.hypothesis.score   = det.score;

            det_msg.results.push_back(hyp);
            det_array.detections.push_back(det_msg); */

            // ───────────────────────────────────
            // Convert YoloDet → vision_msgs format
            // ───────────────────────────────────
            vision_msgs::msg::Detection2D det_msg;

            det_msg.bbox.center.position.x = det.x;
            det_msg.bbox.center.position.y = det.y;
            det_msg.bbox.size_x = det.w;
            det_msg.bbox.size_y = det.h;

            // Fill result array
            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = std::to_string(det.class_id);
            hyp.hypothesis.score    = det.score;    
            // push into message
            det_msg.results.push_back(hyp);
        }

        detections_pub_->publish(det_array);
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
};

// ======================= main =====================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Yolo11TrtNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
