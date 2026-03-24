#ifndef YOLO11_DETECTOR_HPP
#define YOLO11_DETECTOR_HPP
#include "Armor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class YOLO11Detector {

public:
    enum class Camp : bool { Blue = false, Red = true };

public:
    // 构造函数：加载模型并配置 PrePostProcessor
    YOLO11Detector(const std::string& model_path, Camp camp, float conf_threshold = 0.5f, float nms_threshold = 0.2f, const std::string& device = "AUTO")
        : camp_(camp),conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) 
    {
        auto model = core_.read_model(model_path);

        // 使用 OpenVINO 的 PPP (PrePostProcessor) 进行预处理硬件加速
        ov::preprocess::PrePostProcessor ppp(model);
        auto& input = ppp.input();

        // 1. 设置输入 Tensor 的格式： u8, BGR, NHWC排布
        input.tensor()
            .set_element_type(ov::element::u8)
            .set_shape({1, input_height_, input_width_, 3})
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

        // 2. 模型期望的格式： NCHW
        input.model().set_layout("NCHW");

        // 3. 预处理步骤
        // 注意：由于模型 XML 中有 <reverse_input_channels value="YES"/>，模型内部已做 BGR->RGB，
        // 因此这里绝对【不能】再调用 .convert_color(ov::preprocess::ColorFormat::RGB) 导致双重反转！
        input.preprocess()
            .convert_element_type(ov::element::f32)
            .scale(255.0);

        model = ppp.build();

        // 编译模型
        compiled_model_ = core_.compile_model(model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        infer_request_ = compiled_model_.create_infer_request();

        raw_output_ = cv::Mat(50, 8400, CV_32FC1);
        transposed_output_ = cv::Mat(8400, 50, CV_32FC1);
    }

    // 执行推理和后处理
    std::vector<YoloArmor> operator()(const cv::Mat& raw_img) {
        if (raw_img.empty()) return {};

        // 1. Letterbox 图像预处理
        float scale;
        int pad_w, pad_h;
        cv::Mat blob_img = this->letterbox(raw_img, scale, pad_w, pad_h);

// 2. 将数据填充到 Input Tensor (输入已经是零拷贝了，直接绑 blob_img.data)
        ov::Tensor input_tensor(
            ov::element::u8, 
            {1, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_), 3}, 
            blob_img.data
        );
        infer_request_.set_input_tensor(input_tensor);

        // 3. 【加速核心】将提前分配好的 cv::Mat 内存绑定给 Output Tensor
        // 注意：YOLO11 的输出 Shape 是 {1, 50, 8400}
        ov::Tensor output_tensor(
            ov::element::f32, 
            {1, 50, 8400}, 
            raw_output_.data  // <--- 让 OpenVINO 直接写到 cv::Mat 的内存里
        );
        infer_request_.set_output_tensor(output_tensor);

        // 4. 执行推理
        infer_request_.infer();

        // 5. 内存中现在已经有数据了，直接进行转置
        // 转置结果写到 transposed_output_ 里，整个过程无任何 new/malloc 操作
        cv::transpose(raw_output_, transposed_output_);

        // 后面解析的输出改为 transposed_output_
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::vector<cv::Point2f>> all_keypoints;

        // 5. 解析输出
        for (int r = 0; r < this->transposed_output_.rows; r++) {
            // 切片：取分类得分部分
            auto scores = this->transposed_output_.row(r).colRange(4, 4 + class_num_);
            
            double score;
            cv::Point max_point;
            cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

            // 过滤低置信度（提速）
            if (score < conf_threshold_) continue;

            auto xywh = this->transposed_output_.row(r).colRange(0, 4);
            auto kpts = this->transposed_output_.row(r).colRange(4 + class_num_, 4 + class_num_ + 8); // 取8个值(4个点的xy)

            // 还原 Bounding Box 到原图尺寸
            float cx = xywh.at<float>(0);
            float cy = xywh.at<float>(1);
            float w  = xywh.at<float>(2);
            float h  = xywh.at<float>(3);

            int left = static_cast<int>((cx - 0.5 * w - pad_w) / scale);
            int top = static_cast<int>((cy - 0.5 * h - pad_h) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            // 还原关键点到原图尺寸
            std::vector<cv::Point2f> keypoints;
            for (int i = 0; i < 4; i++) {
                float kx = (kpts.at<float>(0, i * 2 + 0) - pad_w) / scale;
                float ky = (kpts.at<float>(0, i * 2 + 1) - pad_h) / scale;
                keypoints.push_back(cv::Point2f(kx, ky));
            }

            boxes.emplace_back(left, top, width, height);
            confidences.emplace_back(static_cast<float>(score));
            class_ids.emplace_back(max_point.x);
            all_keypoints.emplace_back(keypoints);
        }

        // 6. NMS (非极大值抑制)
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, nms_indices);

        std::vector<YoloArmor> results;

        //筛选剔除掉我方阵营装甲板
        for (int idx : nms_indices) 
        {
            int cid = class_ids[idx];
            const std::string& label = class_names_[cid];

            //只识别hero,guard,three和two
            if(cid >= 12) continue;
            
            // 获取目标颜色首字母
            char target_color = label[0]; 


            // 筛选逻辑：
            // 如果我方是 Red (true)，则剔除首字母为 'R' 的目标
            // 如果我方是 Blue (false)，则剔除首字母为 'R' 的目标（这里的逻辑取决于 Camp 的定义）
            
            bool is_friendly = false;
            if (camp_ == Camp::Red) {
                if (target_color == 'R' || target_color == 'P') is_friendly = true;
            } else { // Camp::Blue
                if (target_color == 'B' || target_color == 'P') is_friendly = true;
            }
            
            // 如果是我方单位，跳过不加入结果列表
            if (is_friendly) continue;

            // 同时也建议保留 'E'(灰色/熄灭) 和 'P'(紫色/无敌)，因为它们通常是合法的打击或观测目标
            results.emplace_back(YoloArmor{boxes[idx], confidences[idx], cid, all_keypoints[idx]});
        }
        return results;
    }

    // 可视化函数
    void draw(cv::Mat& img, const std::vector<YoloArmor>& detections) const {
        for (const auto& det : detections) {
            // 生成不同类别对应的颜色
            cv::Scalar color = cv::Scalar(det.class_id * 30 % 255, det.class_id * 60 % 255, det.class_id * 90 % 255);
            
            // 画出检测框
            cv::rectangle(img, det.box, color, 2);
            
            // 画出4个关键点
            for (const auto& kp : det.keypoints) {
                cv::circle(img, kp, 4, cv::Scalar(0, 255, 0), -1);
            }

            // 构造标签文本
            std::string label = (det.class_id < class_names_.size() ? class_names_[det.class_id] : std::to_string(det.class_id)) + 
                                ": " + cv::format("%.2f", det.conf);
            
            // 绘制标签
            int baseLine;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(img, cv::Point(det.box.x, det.box.y - label_size.height - baseLine),
                          cv::Point(det.box.x + label_size.width, det.box.y), color, -1);
            cv::putText(img, label, cv::Point(det.box.x, det.box.y - baseLine),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

private:
    // Letterbox 图像预处理
    cv::Mat letterbox(const cv::Mat& source, float& scale, int& pad_w, int& pad_h) const {
        float scale_x = static_cast<float>(input_width_) / source.cols;
        float scale_y = static_cast<float>(input_height_) / source.rows;
        scale = std::min(scale_x, scale_y);
        
        int new_unpad_w = int(std::round(source.cols * scale));
        int new_unpad_h = int(std::round(source.rows * scale));
        
        pad_w = (input_width_ - new_unpad_w) / 2;
        pad_h = (input_height_ - new_unpad_h) / 2;
        
        cv::Mat resized_img;
        cv::resize(source, resized_img, cv::Size(new_unpad_w, new_unpad_h));
        
        cv::Mat padded_img;
        // 关键修复：使用 114 灰色填充，对齐模型训练时的 padding 策略
        cv::copyMakeBorder(resized_img, padded_img, pad_h, input_height_ - new_unpad_h - pad_h,
                           pad_w, input_width_ - new_unpad_w - pad_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return padded_img;
    }
    Camp camp_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    cv::Mat raw_output_;        // 用于接收 OpenVINO 的原始输出 [50, 8400]
    cv::Mat transposed_output_; // 用于存放转置后的结果 [8400, 50]

    float conf_threshold_;
    float nms_threshold_;
    int input_width_ = 640;
    int input_height_ = 640;
    int class_num_ = 38; // 你的模型有 38 个分类

    const std::vector<std::string> class_names_ = {
        "Bsentry", "Rsentry", "Esentry", "Bone", "Rone", "Eone", "Btwo", "Rtwo", "Etwo", 
        "Bthree", "Rthree", "Ethree", "Bfour", "Rfour", "Efour", "Bfive", "Rfive", "Efive", 
        "Boutpost", "Routpost", "Eoutpost", "Bbase", "Rbase", "Ebase", "Pbase", 
        "Bbasesmall", "Rbasesmall", "Ebasesmall", "Pbasesmall", 
        "Bbalancethree", "Rbalancethree", "Ebalancethree", 
        "Bbalancefour", "Rbalancefour", "Ebalancefour", 
        "Bbalancefive", "Rbalancefive", "Ebalancefive"
    };
};

#endif