#ifndef SOLVER_CLASS_INCLUDE
#define SOLVER_CLASS_INCLUDE
#include "Armor.hpp"
#include "string"
#include "opencv2/opencv.hpp"
#include <deque>
#include <eigen3/Eigen/Core>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/quaternion.hpp>
#include <vector>
class Solver
{
public:

    Solver(std::string config_path);
    
    //解算传入的所有装甲板并返回
    std::vector< std::array<ArmorPosi,2> > operator () (const std::deque<Armor>& armors);
    std::vector< std::array<ArmorPosi,2> > operator () (const std::vector<Armor>& armors);


    std::vector< ArmorPosi > operator () (const std::vector<YoloArmor>& armors);
    
    //解算单个装甲板的位置
    std::array<ArmorPosi,2> operator () (const Armor& armor);

    //坐标系变换
    void ConverToWorld(std::array<ArmorPosi,2>& armor_posis, const cv::Quatd& gripper_to_world);
    void ConverToWorld(std::vector< std::array<ArmorPosi,2> >& armors_posis, const cv::Quatd& gripper_to_world);

    void ConverToWorld(ArmorPosi& armor_posi, const cv::Quatd& gripper_to_world);
    void ConverToWorld(std::vector<ArmorPosi>& armors_posi, const cv::Quatd& gripper_to_world);    

    //筛选（去掉位置不正确的装甲板）
    void Filter(std::vector< std::array<ArmorPosi,2> >& armors_posis,
                std::vector<cv::Mat>& armors_pattern,
                const cv::Quatd& gripper_to_world,
                const Eigen::Matrix<double, 3, 1>& Gun,
                const size_t num = 1);
                
    void FilterAndConverToWorld(std::vector<ArmorPosi>& armors_posi,
                const cv::Quatd& gripper_to_world,
                const Eigen::Matrix<double, 3, 1>& Gun,
                const size_t num = 1);

    void ansShow(const cv::Point3d& posi,cv::Mat& image);
    void ansShow(const ArmorPosi& armor,cv::Mat& image);

private:
    cv::Mat_<double> cameraMatrix;
    cv::Mat_<double> distCoeffs;

    cv::Mat_<double> R_Cam_to_gripper;
    cv::Mat_<double> T_Cam_to_gripper;

    cv::Mat_<double> BigArmorCenter;
    cv::Mat_<double> SmallArmorCenter;
    std::vector<cv::Point3f> objectBigArmorP{{0,0,0},{23,0,0},{23,5.5,0},{0,5.5,0}};
    std::vector<cv::Point3f> objectSmallArmorP{{0,0,0},{13.5,0,0},{13.5,5.5,0},{0,5.5,0}};
};

#endif