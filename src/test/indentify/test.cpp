#include "../../rm-main/include/HikCamera.hpp"
#include "../../rm-main/include/Detector.hpp"
#include "../../rm-main/include/Armor.hpp"
#include "../../rm-main/include/Solver.hpp"
#include "../../rm-main/include/NumClassifier.hpp"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <deque>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
static int num=0;
static int Armor_num=0;
static std::chrono::duration<double> total_elapsed_seconds(0.0);
static std::chrono::duration<double> delay_seconds(0.0);
static std::chrono::duration<double> total_delay_seconds(0.0);
static auto total = std::chrono::nanoseconds(0);
int main()
{
    Detector detect(Light::Color::Red,0.5);
    NumClassifier classifier("../../../rm-main/model/mobilenet_v3_arcface_best.onnx","../../../rm-main/model/centers.yaml");

    Solver Sov("../../../config/Solver_config.yaml");
    io::HikCamera Hik(2,17);
    Hik.continueCap(5);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    // cv::namedWindow("gray_img",cv::WINDOW_NORMAL);
    while(true)
    {
        io::HikCamera::ImageData frame; 
            
        Hik.read(frame);

        if(frame.image.empty()) continue;
        auto start = std::chrono::steady_clock::now();
        std::vector<cv::Mat> armors_pattern;
        auto armors = detect(frame.image, armors_pattern);

        auto armorsposis = Sov(armors);
        auto armorsposi = classifier(armorsposis,armors_pattern);
        if(!armors.empty())
        {
            total += std::chrono::steady_clock::now() - start;
            num++;
            if(num%100 == 0&& num != 0)
            {
                std::cout<<(total.count()/(float)num)*1e-6<<"\n";
                num = 0;
                total = std::chrono::nanoseconds(0);
            }
        }


        std::cout<<"Armor num: "<<armors.size()<<"\n";

        if(!armorsposi.empty()) 
        {
            for(int i = 0;i < armorsposi.size();i++)
            {
                std::cout<<"ID: "<<static_cast<int>(armorsposi[i].type)<<" confidence: "<< armorsposi[i].confidence<<"\n";
                Sov.ansShow( armorsposi[i], frame.image);
            }
            std::cout<<"-----------------------------------------------------------\n";
        }
        cv::imshow("hh",frame.image);
        cv::waitKey(1);
        if(armorsposi.empty()) continue;
        int Id = static_cast<int>(armorsposi[0].type);
        std::cout<<"ID: "<<Id<<" confidence: "<< armorsposi[0].confidence<<"\n";

    }
}
    // std::string config_path = "/home/king/desktop/SinAim_rm/10.16/config/Solver_config.yaml";

    // cv::FileStorage fs;
    // if (!fs.open(config_path, cv::FileStorage::READ)) 
    //     std::cerr << "Error: Failed to open YAML file: " << config_path << std::endl;
    
    // std::cout << "Successfully opened " << config_path << std::endl;
    // std::cout << "------------------------------------------" << std::endl;

    // cv::Mat_<double> cameraMatrix;
    // cv::Mat_<double> distCoeffs;
    // fs["camera_matrix"] >> cameraMatrix;
    // fs["distortion_coeffs"] >> distCoeffs;

    // for(int i=0;i<1;i++)
    // {
    //     for(int j=0;j<5;j++)
    //     {
    //             std::cout<<distCoeffs(i,j)<<" ";
    //     }
    //     std::cout<<std::endl;
    // }