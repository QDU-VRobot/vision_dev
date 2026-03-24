#ifndef DETECTOR_CLASS
#define DETECTOR_CLASS
#include "Armor.hpp"
#include <deque>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>
class Detector {
public:
    Detector(Light::Color color,float confidence);
    
    std::deque<Armor> operator () (cv::Mat& frame,std::vector<cv::Mat>& armors_pattern);
    std::deque<Armor> operator () (cv::Mat& frame,std::vector<cv::Mat>& armors_pattern,bool isSmallROI);

    void ArmorShow(cv::Mat & rgb_img, const std::deque<Armor> & armors);
    void ArmorShow(cv::Mat & rgb_img, const std::vector<Armor> & armors);

public:
    float confidence;
    Light::Color color;
    cv::Mat gray_img;
    cv::Mat rgb_img;
    
public:
    cv::Mat preprocessImage(cv::Mat& rgb_img); //图像预处理
    std::deque<Light> FindLight(const cv::Mat & binary_img); //寻找灯条
    std::deque<Armor> FindArmor(const std::deque<Light> & lights); //寻找装甲板
    std::vector<cv::Mat> ROIArmor(const std::deque<Armor>& armors);
    std::vector<cv::Mat> SmallROIArmor(const std::deque<Armor> & armors);
};
#endif