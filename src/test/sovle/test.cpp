#include "../../rm-main/include/HikCamera.hpp"
#include "../../rm-main/include/Detector.hpp"
#include "../../rm-main/include/Armor.hpp"
#include "../../rm-main/include/Solver.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <deque>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <vector>

static int num=0;
static std::chrono::duration<double> total_elapsed_seconds(0.0);
int main()
{
    io::HikCamera Hik(1,10);
    Hik.continueCap(5);
    cv::namedWindow("hh");
    cv::namedWindow("result");
    cv::namedWindow("debug");

    Detector detect(Light::Color::Blue,0.5,"/home/king/AUTO-Aming-system/rm-main/model/mobilenet_v3_112_rgb.onnx");
    Solver Sov("/home/king/AUTO-Aming-system/config/Solver_config.yaml");
    
    while(true)
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        io::HikCamera::ImageData frame; 
        Hik.read(frame);
        // frame.image = cv::imread("/home/king/Desktop/SinAim_rm/10.15.13/workindentify/images/frame2.png");
        cv::Mat debug = detect.preprocessImage(frame.image);

        
        std::vector<Armor> armors = detect(frame.image);
        std::vector<ArmorPosi> armors_posi = Sov(armors);


        //可视化装甲板中心
        if(!armors_posi.empty())
        {
            
            Sov.ansShow(armors_posi[0].posi, frame.image);
            if(num%30==0&&num!=0) 
            {
                std::cerr<<"----------------------------------------\n"<<
                cv::norm(armors_posi[0].posi)/10<<
                armors_posi[0].posi/10
                <<"\n";
            }
        }
        // detect.ArmorShow(frame.image, armors);
        // cv::imshow("hh",frame.image);
        // cv::imshow("result",frame.image);
        // cv::imshow("debug",debug);
        // cv::waitKey(1);

                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        
        total_elapsed_seconds += end - start;
        num++;

        if(num%100==0&&num!=0)
        {
            std::cout<<total_elapsed_seconds.count()/num<<"\n"
            <<num/total_elapsed_seconds.count()<<" fps\n";

        }
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