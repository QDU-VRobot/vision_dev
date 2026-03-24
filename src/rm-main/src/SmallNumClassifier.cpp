#include "../include/SmallNumClassifier.hpp"
#include <opencv2/core/types.hpp>
#include <vector>


SmallNumClassifier::SmallNumClassifier(std::string model_path)
{
    Net = cv::dnn::readNetFromONNX(model_path);
    // 设置首选的计算后端为 OpenVINO Inference Engine
    // Net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    // 设置首选的计算目标设备为 CPU
    Net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // Create blob from image

}
std::vector<SmallNumClassifier::Ans> SmallNumClassifier::Classify(const std::vector<cv::Mat>& armors_pattern)
{
    std::vector<SmallNumClassifier::Ans> ans;
    if(armors_pattern.empty()) return ans;
    ans.reserve(armors_pattern.size());

    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImages(armors_pattern, blob);

     // Set the input blob for the neural network
    this->Net.setInput(blob);
    // Forward pass the image blob through the model
    cv::Mat outputs = this->Net.forward();

    //读取结果
    for (int i = 0; i < outputs.rows; ++i) 
    {
        // 获取第 i 张图片对应的得分行
        cv::Mat scores = outputs.row(i);

        // Do softmax
        float max_prob = *std::max_element(scores.begin<float>(), scores.end<float>()); //max_element函数返回的是一个迭代器，要获取实际值，需要解引用*
        cv::Mat softmax_prob;
        cv::exp(scores - max_prob, softmax_prob);
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
        softmax_prob /= sum;

        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
        int label_id = class_id_point.x;
        ans.emplace_back(label_id,confidence);
    }
    return ans;
}
std::vector<ArmorPosi> SmallNumClassifier::operator()(std::vector< std::array<ArmorPosi,2> >& armors,const std::vector<cv::Mat>& armors_pattern)
{
    std::vector<ArmorPosi> result;
    if(armors.empty()) return result;

    result.reserve(armors.size());

    std::vector<SmallNumClassifier::Ans> ans = Classify(armors_pattern);
    
    for(int i = 0;i < ans.size();i++)
    {
        if(ans[i].id == 7) continue ;
        
        if(ans[i].confidence > 0.70)
        {
            if(ans[i].id == 0 || ans[i].id == 1)
            {
                result.emplace_back(armors[i][1]);
                result.back().type = static_cast<ArmorPosi::Type>(ans[i].id);
                result.back().confidence = ans[i].confidence;
            } 
                
            else 
            {
                result.emplace_back(armors[i][0]);
                result.back().type = static_cast<ArmorPosi::Type>(ans[i].id);
                result.back().confidence = ans[i].confidence;
            } 
                
        }
    }
    
    return result;
}