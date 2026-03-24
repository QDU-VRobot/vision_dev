#include "../../rm-main/include/HikCamera.hpp"
#include "../../rm-main/include/Detector.hpp"
#include "../../rm-main/include/Armor.hpp"
#include "../../rm-main/include/Solver.hpp"
#include "../../rm-main/include/NumClassifier.hpp"
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
    io::HikCamera Hik(7,17);
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


        if(!armorsposi.empty()) detect.ArmorShow(frame.image, armors);
        cv::imshow("hh",frame.image);
        cv::waitKey(1);

        if(armorsposi.empty()) continue;
        int Id = static_cast<int>(armorsposi[0].type);
        std::cout<<"ID: "<<Id<<" confidence: "<< armorsposi[0].confidence<<"\n";

        cv::imshow("Roi",armors_pattern[0]);
        auto key = cv::waitKey(0);
        if(key == ' ' )
        {
            cv::imwrite("../data/image_"+std::to_string(num++)+".png", armors_pattern[0]);
        }           

    }
}


































// int main()
// {
//     Detector detect(Light::Color::Blue,0.3);
//     io::HikCamera Hik(7,17);
//     Hik.continueCap(5);

    
//     std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

//     // cv::namedWindow("gray_img",cv::WINDOW_NORMAL);
//     while(true)
//     {
//         io::HikCamera::ImageData frame; 

//         Hik.read(frame);
        
//         if(frame.image.empty()) continue;

//         detect.rgb_img = frame.image;
//         cv::Mat gray_img = detect.preprocessImage(frame.image);


        
//         auto lights  = detect.FindLight(gray_img);
//         #ifdef Debug
//         std::cout <<"lights num:" << lights.size() << "\n";
//         #endif
//         auto armors = detect.FindArmor(lights);
//         #ifdef Debug
//         std::cerr <<"armors num:" << armors.size() << "\n";
//         #endif
//         auto roi = detect.ROIArmor( armors );


//         detect.ArmorShow(frame.image, armors);
//         cv::imshow("gray_img",frame.image);
//         cv::waitKey(1);


//     }
// }
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