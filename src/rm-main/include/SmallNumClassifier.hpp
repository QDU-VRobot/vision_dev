#ifndef INCLUDE_SMALLNUMBER_CLASSIFIER_HPP
#define INCLUDE_SMALLNUMBER_CLASSIFIER_HPP

// OpenCV
#include <opencv2/opencv.hpp>
#include "Armor.hpp"

// STL
#include <string>
#include <vector>

class SmallNumClassifier
{
public:
    struct Ans{
        int id;
        float confidence;
        Ans(int id,double con):id(id),confidence(con){}
    };
    SmallNumClassifier(std::string model_path);
    std::vector<ArmorPosi> operator()(std::vector< std::array<ArmorPosi,2> >& armors,const std::vector<cv::Mat>& armors_pattern);
    std::vector<Ans> Classify(const std::vector<cv::Mat>& armors_pattern);

private:
    cv::dnn::Net Net;
};




#endif