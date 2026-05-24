#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#if ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_base.hpp"
#include "armor_detector/light_corner_corrector.hpp"

#if ARMOR_DETECTOR_HAS_TENSORRT
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "armor_detector/gpu_preprocessor.hpp"

#elif ARMOR_DETECTOR_HAS_OPENVINO
#include <openvino/openvino.hpp>
#endif  // ARMOR_DETECTOR_HAS_OPENVINO / ARMOR_DETECTOR_HAS_TENSORRT

namespace rclcpp
{
class Node;
}

namespace rm_auto_aim
{

class YoloDetector : public DetectorBase
{
 public:
  struct YoloParams
  {
    std::string model_path;
    std::string device = "CPU";
    int input_size = 640;
    float score_threshold = 0.7f;
    float min_confidence = 0.8f;
    float nms_threshold = 0.3f;
    std::vector<std::string> ignore_classes;
    int detect_color;
    int num_keypoints = 4;
    bool end_to_end = false;

    // 合法性判别 + SMALL/LARGE 分型参数 (对齐传统路径 Detector::IsArmor):
    //   1) 两灯条长度比 (短/长) > min_light_ratio
    //   2) 中心距 / 平均灯条长 落在 [min_small, max_small) ∪ [min_large, max_large) 之一
    //   3) 两灯条中心连线与水平方向夹角 < max_armor_angle
    // 通过合法性判别后, 以 min_large_center_distance 作为 SMALL/LARGE 分界。
    double min_light_ratio = 0.7;
    double min_small_center_distance = 0.8;
    double max_small_center_distance = 3.2;
    double min_large_center_distance = 3.2;
    double max_large_center_distance = 5.5;
    double max_armor_angle = 35.0;

    // 灯条二值化 + 颜色过滤 (对齐传统路径 Detector::PreprocessImage / FindLights):
    //   在 YOLO 解析出关键点并构建灯条后, 用同一套灰度阈值再做一次二值化, 并基于
    //   contour 的 R/B 像素和判定颜色, 用于过滤模型对不亮灯条的假阳性检测。
    int binary_lower_thres = 160;
    int binary_upper_thres = 255;

    // 灯条角点校正 (对齐传统路径 Detector::CornerCorrectorParams):
    //   在 PostProcessLights 完成验证后, 利用灰度图 + PCA 沿对称轴搜索亮度梯度,
    //   把 YOLO 关键点细化到亚像素级灯条端点。
    bool use_corner_corrector = false;
    double cc_max_brightness = 25.0;
    double cc_scale = 0.07;
    double cc_start = 0.4;
    double cc_end = 0.6;
  };

  static std::unique_ptr<YoloDetector> Create(rclcpp::Node& node);

  explicit YoloDetector(const YoloParams& params);

  ~YoloDetector() override;
  YoloDetector(const YoloDetector&) = delete;
  YoloDetector& operator=(const YoloDetector&) = delete;

  DetectionResult Detect(const cv::Mat& rgb_img) override;

  void DrawResults(cv::Mat& img) override;

 private:
  std::vector<Armor> Parse(double scale, const cv::Mat& output);
#if ARMOR_DETECTOR_HAS_TENSORRT
  std::vector<Armor> ParseEnd2End(double scale);

  void InitTrtRaw();
  void InitTrtEnd2End();
  DetectionResult DetectTrtRaw(const cv::Mat& rgb_img);
  DetectionResult DetectTrtEnd2End(const cv::Mat& rgb_img);
#elif ARMOR_DETECTOR_HAS_OPENVINO
  std::vector<Armor> ParseOpenVinoEnd2End(double scale, int n, const float* scores,
                                          const int* classes, const float* kpts,
                                          int kpt_channels);

  // 根据输入图像分辨率刷新 letterbox 比例缓存与持久 input buffer 的 padding 区。
  // 仅当输入分辨率变化时才会执行实际工作。
  void RefreshLetterboxCache(int rows, int cols);
#endif
  void SortKeypoints(std::vector<cv::Point2f>& keypoints);
  // 装甲板合法性判别 + SMALL/LARGE 分型 (对齐传统路径 Detector::IsArmor)。
  // 不合法时返回 ArmorType::INVALID, 调用方应在解析阶段直接 continue。
  ArmorType DetermineArmorType(const Light& light_1, const Light& light_2);

  // YOLO 解析完成后的后处理:
  //   1) 对原图灰度化 + 阈值二值化 (复用传统路径同名参数)
  //   2) 对每个 armor 的两根灯条, 在二值化图上找最近 contour, 用 R/B 像素和重新
  //      判定颜色; 任一灯条未点亮或颜色不匹配, 整个 armor 被丢弃
  //   3) 用 contour 的 minAreaRect 回填 RotatedRect 基类与 width, 这样 width > 3
  //      时 LightCornerCorrector::CorrectCorners 才会真正生效
  //   4) 如启用 use_corner_corrector, 调用 LightCornerCorrector 对灯条端点做亚像素
  //      级修正
  void PostProcessLights(const cv::Mat& rgb_img);

  // 预计算的 per-class LUT, 避免每帧字符串比较 / map_label / std::find。
  // 由构造函数在配置完 ignore_classes 后一次性建立, 之后只读。
  std::vector<std::string> class_label_lut_;     // raw_label → mapped label
  std::vector<int> class_color_lut_;             // RED / BLUE / -1 (颜色无关)
  std::vector<std::uint8_t> class_ignored_lut_;  // 1=该 class 命中 ignore_classes

  YoloParams params_;
  int class_num_;

#if ARMOR_DETECTOR_HAS_TENSORRT
  std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;

  std::string trt_input_name_;
  std::string trt_output_name_;

  nvinfer1::Dims trt_input_dims_{};
  nvinfer1::Dims trt_output_dims_{};

  void* trt_d_input_ = nullptr;
  void* trt_d_output_ = nullptr;
  cudaStream_t trt_stream_ = nullptr;

  std::size_t trt_input_bytes_ = 0;
  std::size_t trt_output_bytes_ = 0;

  std::vector<float> trt_host_output_;

  // IO 张量名
  std::string in_name_;           // images / input
  std::string out_num_name_;      // num_dets     [1, 1]        int32
  std::string out_boxes_name_;    // det_boxes    [1, K, 4]     fp32  (xyxy, 640 尺度)
  std::string out_scores_name_;   // det_scores   [1, K]        fp32
  std::string out_classes_name_;  // det_classes  [1, K]        int32
  std::string out_kpts_name_;     // det_kpts     [1, K, K*D]   fp32  (640 尺度)

  // Device / pinned host buffers
  void* d_input_ = nullptr;
  int* d_num_ = nullptr;
  float* d_boxes_ = nullptr;
  float* d_scores_ = nullptr;
  int* d_classes_ = nullptr;
  float* d_kpts_ = nullptr;

  int* h_num_ = nullptr;
  float* h_boxes_ = nullptr;
  float* h_scores_ = nullptr;
  int* h_classes_ = nullptr;
  float* h_kpts_ = nullptr;

  cudaStream_t stream_ = nullptr;

  // 形状元数据
  int keep_topk_ = 0;     // engine 里 NMS 的 max_output_boxes
  int kpt_channels_ = 0;  // num_kpts * kp_dim (2 或 3)
  std::size_t input_bytes_ = 0;

  // CUDA Graph
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_ready_ = false;

  std::unique_ptr<GpuPreprocessor> preprocessor_;

  cudaEvent_t ev_start_ = nullptr;
  cudaEvent_t ev_end_ = nullptr;

#elif ARMOR_DETECTOR_HAS_OPENVINO
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

  // 持久化 input buffer。直接 wrap OpenVINO infer_request 的内置 input tensor,
  // 避免每帧 cv::Mat 分配 + cv::Scalar 全图清零 + ov::Tensor 构造 + set_input_tensor。
  ov::Tensor ov_input_tensor_;
  cv::Mat ov_input_mat_;

  // letterbox 比例缓存。相机分辨率不变时 (实际场景 99% 都是) 这些值整轮复用。
  int ov_last_rows_ = 0;
  int ov_last_cols_ = 0;
  double ov_cached_scale_ = 0.0;
  int ov_cached_w_ = 0;
  int ov_cached_h_ = 0;
#endif  // ARMOR_DETECTOR_HAS_OPENVINO / ARMOR_DETECTOR_HAS_TENSORRT

  std::vector<Armor> last_armors_;

  // 灯条后处理状态: gray/binary 缓存以及可选的角点校正器
  std::unique_ptr<LightCornerCorrector> light_corner_corrector_;
  cv::Mat gray_img_;
  cv::Mat binary_img_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_