#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <memory>
#include <vector>
#include <numeric>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

class DepthAnythingTrtNode : public rclcpp::Node {
public:
    explicit DepthAnythingTrtNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : Node("depth_anything_trt_node", options)
    {
        camera_topic_ = declare_parameter("camera_topic", "/camera/image_raw");
        engine_path_  = declare_parameter("engine_path", "/home/mecatron/depth_anything_v2_vits_fp16.engine");
        output_topic_ = declare_parameter("output_topic", "/depth/image_raw");

        input_w_     = declare_parameter("input_width", 518);
        input_h_     = declare_parameter("input_height", 518);
        max_depth_m_ = declare_parameter("max_depth_m", 1.275f);

        // === CALIBRATION MODE ===
        do_calibration_   = declare_parameter("do_calibration", false);
        target_distance_m_ = declare_parameter("target_distance_m", 1.0f);
        calibration_samples_ = declare_parameter("calibration_samples", 30);

        RCLCPP_INFO(get_logger(), "Loading engine: %s", engine_path_.c_str());

        if (!loadEngine(engine_path_)) {
            throw std::runtime_error("Failed to load TensorRT engine");
        }

        auto qos = rclcpp::SensorDataQoS();

        sub_ = create_subscription<sensor_msgs::msg::Image>(
            camera_topic_,
            qos,
            std::bind(&DepthAnythingTrtNode::imageCb, this, std::placeholders::_1)
        );

        pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, qos);

        RCLCPP_INFO(get_logger(), "TensorRT DepthAnything Node Ready");
    }

    ~DepthAnythingTrtNode()
    {
        if (input_dev_) cudaFree(input_dev_);
        if (output_dev_) cudaFree(output_dev_);
    }

private:

    // === TENSORRT OBJECTS ===
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // IO Buffers
    float *input_dev_{nullptr};
    float *output_dev_{nullptr};

    std::string input_name_;
    std::string output_name_;
    int out_h_{0};
    int out_w_{0};

    // Parameters
    std::string camera_topic_;
    std::string output_topic_;
    std::string engine_path_;
    int input_w_, input_h_;
    float max_depth_m_;

    bool invert_depth_ = false;
    int last_pad_x_=0, last_pad_y_=0, last_new_w_=0, last_new_h_=0;

    // ROS
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

    // Smoothing
    cv::Mat accumulated_;
    float alpha_ = 0.1f;

    // === CALIBRATION VARIABLES ===
    bool do_calibration_ = false;
    float target_distance_m_ = 1.0f;
    int calibration_samples_ = 30;

    int collected_samples_ = 0;
    std::vector<float> depth_samples_;

    // -------------------------------------------
    // LOAD ENGINE
    // -------------------------------------------
    bool loadEngine(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;

        size_t size = f.tellg();
        f.seekg(0, std::ios::beg);

        std::vector<char> data(size);
        f.read(data.data(), size);

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) return false;

        engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
        if (!engine_) return false;

        context_.reset(engine_->createExecutionContext());
        if (!context_) return false;

        input_name_  = engine_->getIOTensorName(0);
        output_name_ = engine_->getIOTensorName(1);

        auto out_dims = engine_->getTensorShape(output_name_.c_str());
        out_h_ = out_dims.d[1];
        out_w_ = out_dims.d[2];

        cudaMalloc(&input_dev_,  3 * input_h_ * input_w_ * sizeof(float));
        cudaMalloc(&output_dev_, out_h_ * out_w_ * sizeof(float));

        context_->setTensorAddress(input_name_.c_str(), input_dev_);
        context_->setTensorAddress(output_name_.c_str(), output_dev_);

        return true;
    }


    // -------------------------------------------
    // CALLBACK
    // -------------------------------------------
    void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        // PREPROCESSING
        cv_bridge::CvImageConstPtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvShare(msg, "bgr8"); }
        catch (...) { return; }

        cv::Mat img = cv_ptr->image;
        if (img.empty()) return;

        int orig_w = img.cols, orig_h = img.rows;
        int model_w = input_w_, model_h = input_h_;

        float scale = std::min(
            model_w / float(orig_w),
            model_h / float(orig_h)
        );

        int new_w = int(orig_w * scale);
        int new_h = int(orig_h * scale);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h));

        int pad_x = (model_w - new_w) / 2;
        int pad_y = (model_h - new_h) / 2;

        last_pad_x_ = pad_x;
        last_pad_y_ = pad_y;
        last_new_w_ = new_w;
        last_new_h_ = new_h;

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded,
            pad_y, model_h - new_h - pad_y,
            pad_x, model_w - new_w - pad_x,
            cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        padded.convertTo(padded, CV_32FC3, 1.0/255.0);
        cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

        std::vector<cv::Mat> ch(3);
        cv::split(padded, ch);

        int cs = model_w * model_h;
        std::vector<float> input_host(3 * cs);

        memcpy(input_host.data(),          ch[0].ptr<float>(), cs * sizeof(float));
        memcpy(input_host.data() + cs,     ch[1].ptr<float>(), cs * sizeof(float));
        memcpy(input_host.data() + 2*cs,   ch[2].ptr<float>(), cs * sizeof(float));

        cudaMemcpy(input_dev_, input_host.data(),
                   input_host.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        if (!context_->enqueueV3(0)) return;

        std::vector<float> out_host(out_h_ * out_w_);
        cudaMemcpy(out_host.data(), output_dev_,
                   out_host.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cv::Mat depth_raw(out_h_, out_w_, CV_32FC1, out_host.data());
        
        cv::Mat safe_raw;
        cv::add(depth_raw, cv::Scalar(1e-6), safe_raw); // Avoid divide by 0
        
        cv::Mat depth_m;
        // Formula: Depth = Calibration_Constant / Raw_Value
        cv::divide(max_depth_m_, safe_raw, depth_m);

        // RESIZE & UNPAD
        cv::Rect roi(last_pad_x_, last_pad_y_, last_new_w_, last_new_h_);
        cv::Mat depth_unpadded = depth_m(roi).clone();
        
        cv::Mat depth_resized;
        cv::resize(depth_unpadded, depth_resized, img.size());

        // THE "INVISIBLE" FILTER (NaN Masking)
        for (int r = 0; r < depth_resized.rows; r++) {
            float* row_ptr = depth_resized.ptr<float>(r);
            for (int c = 0; c < depth_resized.cols; c++) {
                float d = row_ptr[c];
                // CONDITION: If closer than 10cm OR further than 3.0m
                if (d < 0.1f || d > 3.0f) {
                    // Set to NaN (Invisible in Rviz)
                    row_ptr[c] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        // === CALIBRATION MODE =============================
        if (do_calibration_) {
            // Sample from the RESIZED image, not the raw tensor
            float center_d = depth_resized.at<float>(img.rows/2, img.cols/2);

            // Sanity check: Don't calibrate on NaNs
            if (std::isnan(center_d)) {
                RCLCPP_WARN(get_logger(), "[CALIB] Center pixel is NaN (Too close/far)!");
                return;
            }

            depth_samples_.push_back(center_d);
            collected_samples_++;

            RCLCPP_INFO(get_logger(),
                "[CALIB] Sample %d/%d → depth_center=%.3f m",
                collected_samples_, calibration_samples_, center_d
            );

            if (collected_samples_ >= calibration_samples_) {
                float avg = std::accumulate(depth_samples_.begin(), depth_samples_.end(), 0.0f)
                          / depth_samples_.size();

                float scale = target_distance_m_ / avg;

                RCLCPP_WARN(get_logger(),
                    "\n=====================================\n"
                    " CALIBRATION COMPLETE\n"
                    " avg_measured = %.3f m\n"
                    " target_dist  = %.3f m\n"
                    " ➤ Recommended max_depth_m = %.3f\n"
                    "=====================================\n",
                    avg, target_distance_m_, scale
                );

                do_calibration_ = false; 
            }
            return;
        }
        
        // NORMAL MODE — Smoothing + Publish
        if (accumulated_.empty()) {
            accumulated_ = depth_resized.clone();
        } else {
            cv::addWeighted(depth_resized, alpha_, accumulated_, 1.0-alpha_, 0.0, accumulated_);
        }

        auto out_msg = cv_bridge::CvImage(msg->header, "32FC1", accumulated_).toImageMsg();
        pub_->publish(*out_msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthAnythingTrtNode>());
    rclcpp::shutdown();
    return 0;
}
