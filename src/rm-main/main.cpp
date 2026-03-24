#include "include/HikCamera.hpp"
#include "include/RTSerial.hpp"
#include "include/Yolo.hpp"
#include "include/fastqueue.hpp"
#include "include/Detector.hpp"
#include "include/Solver.hpp"
#include "include/Shooter.hpp"
#include "include/Target.hpp"
#include "include/Tracker.hpp"
#include "include/Data.hpp"
#include "Function.hpp"
#include "include/RerunVisualizer.hpp"

#include <chrono>
#include <cstdio>
#include <eigen3/Eigen/src/Core/MatrixBase.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>

#define SmallMainDebug
#ifdef SmallMainDebug
double R_sum = 0.0;
int R_count = 0;
#endif

//性能测试工具
struct Test
{
    int num = 0;
    std::chrono::nanoseconds total{0};

    void count(const std::chrono::nanoseconds& time);
    void clear();
    void show();
};

static FastQueue<FrameData> Frames(10);

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

io::HikCamera Hik(1,16);
io::RTSerial<Packet> ser(80);

// 传统视觉检测器
Detector detect(Light::Color::Blue, 0.4);

// YOLO检测器
YOLO11Detector yolo11detect("../model/yolo11.xml", YOLO11Detector::Camp::Blue);

RerunVisualizer viz("RoboMaster_AutoAim");

Solver Sov("../../config/Solver_config.yaml");

// 追踪器
Tracker track;

Shooter shoot(Eigen::Matrix<double,3,1>(0.9999283656297303,
 0.002822337907989043,
 -0.01163176761242803));

Test test;

int main() {

    //1.0初始化串口
    std::cout<<sizeof(Packet)<<std::endl;

    std::function<bool(const Packet&)> check_fuc = io::CRC8::Check<Packet>;
    ser.setCheckfuc(check_fuc);
    int ret = ser.openDevice("/dev/ttyACM0", 460800);

    if(ret == 1)
        std::cout<<"serial open ok"<<"\n";
    else
        std::cerr<<"serial open err: "<<ret<<"\n";

    ser.startReceive(100);


    //2.0初始化相机
    Hik.continueCap(3);

    //3.0创建数据配对线程，并将数据发布到Frames环形队列
    std::thread match_thread = std::thread(rm::IMUAndImageMatchFunction, std::ref(Hik), std::ref(ser), std::ref(Frames));

    // cv::namedWindow("frame");
        auto start = std::chrono::steady_clock::now();
    std::printf("Start main loop\n");

    while(true)
    {
        if(Frames.empty()) continue;

        FrameData frame;

        while (!Frames.empty())
        {  
            Frames.pop(frame);
        }
        if(frame.image.empty()) continue;

        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);
        
                // 性能统计
        // test.count(std::chrono::steady_clock::now() - start);
        // start = std::chrono::steady_clock::now();

        // if(test.num%200 == 0 && test.num != 0) {
        //     test.show();
        //     test.clear();
        // }
        //计算枪管方向

        const Eigen::Matrix<double, 3, 1>& Gun = shoot.GunDirection(Eigen::Quaterniond{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z});

    //   auto t1_ = std::chrono::steady_clock::now();
        // 1. 传统视觉检测
        std::vector<cv::Mat> armors_pattern;
        auto opencv_armors = detect(frame.image, armors_pattern);

        // std::cout<<"opencv_armors num:" << opencv_armors.size() << "\n";
  
        // 2. YOLO检测
        std::vector<YoloArmor> yolo_armors = yolo11detect(frame.image);

        // std::cout<<"yolo_armors num:" << yolo_armors.size() << "\n";

        // 3. 融合传统视觉和YOLO的结果
        auto fused_yolo_armors = rm::MatchYoloAndOpenCV(opencv_armors, yolo_armors);

        // yolo11detect.draw(frame.image, fused_yolo_armors);
        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);
        // std::cout<<"fused_yolo_armors num:" << fused_yolo_armors.size() << "\n";

        // 4. 解算装甲板位置 (使用融合后的YOLO结果)
        std::vector<ArmorPosi> armors_posi = Sov(fused_yolo_armors);

        // 5. 筛选装甲板，内部自动坐标系转换到世界坐标系
        Sov.FilterAndConverToWorld(armors_posi, frame.quat, Gun, 20);
        // std::cout<<"time: " << (std::chrono::steady_clock::now()-t1_).count() <<"\n";
        // std::cout<<"FilterAndConverToWorld armors_posi num:" << armors_posi.size() << "\n";
        
        // 7. 使用Tracker进行追踪
        track(armors_posi, frame.quat, Gun, rm::SolveDt(next_point, frame.time, 0.01));
        next_point = frame.time;

        // 8. 获取当前追踪的机器人
        Robot* current_robot = track.getCurrentRobot();

        if (current_robot == nullptr)
        {
            // 没有追踪到目标，跳过
            continue;
        }

        // 9. 预测和瞄准
        double dt = shoot.FlyTime(current_robot->center);
        Eigen::Matrix<double, 4, 4> aims = current_robot->Predic(dt);

        Eigen::Matrix<double, 4, 1> aim = rm::ChooseBestAimArmor(aims, current_robot->Speed, Gun);

        std::cout<<aim<<"\n";
        // std::cout<<aim.norm()<<"\n";
        // 10. 计算射击角度
        std::array<double, 2> Pitch_and_Yaw = shoot(aim.block<3,1>(0,0));

        // 11. 发送控制指令
        bool fire_ = rm::CheckFireCondition(frame.quat, Pitch_and_Yaw, aim, Gun, 0.03, 0.03);
        rm::SendMessageToRobot(ser, Pitch_and_Yaw[0], Pitch_and_Yaw[1], true);
        // std::cout<<fire_<<"\n";
        // std::cout<<"Pitch: "<<Pitch_and_Yaw[0]-0.05<<" Yaw: "<<Pitch_and_Yaw[1]<<"\n";
        // 性能统计
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {
            test.show();
            test.clear();
        }
        #ifdef SmallMainDebug
            viz.update(*current_robot, aims, dt, Gun);
        #endif
        // 可视化
        // for(const auto& armor_posi : armors_posi)
        // {
        //     Sov.ansShow(armor_posi.posi, frame.image);
        // }
        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);
    }


    return 0;
}




//性能测试工具
void Test::count(const std::chrono::nanoseconds& time)
{
    this->num++;
    total += time;
}

void Test::clear()
{
    this->num = 0;
    this->total = std::chrono::nanoseconds(0);
}

void Test::show()
{
    std::cout<< this->num / ( ( (double)this->total.count() ) * 1e-9 )<< "Hz\n" ;
}