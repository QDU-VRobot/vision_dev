#ifndef ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_
#define ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_

// OpenCV
#include "Armor.hpp"
#include <array>
#include "eigen3/Eigen/Core"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
// STL
#include <string>
#include <vector>

class NumClassifier
{
public:
    struct Ans{
        int id;
        float confidence;
        Ans(int id,double con):id(id),confidence(con){}
    };
    NumClassifier(std::string model_path,std::string yaml_path);
    std::vector<ArmorPosi> operator()(std::vector< std::array<ArmorPosi,2> >& armors,const std::vector<cv::Mat>& armors_pattern);
    std::vector<Ans> Classify(const std::vector<cv::Mat>& armors_pattern);
    
private:
    bool has_cpu = false;
    bool has_gpu = false;

    ov::Core core;

    ov::CompiledModel compiled_model_GPU, compiled_model_CPU;

    std::vector<ov::InferRequest> infer_request_GPU, infer_request_CPU;

    static Eigen::Matrix<float, 7, 128> centers;
};




#endif