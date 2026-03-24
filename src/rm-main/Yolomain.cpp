#include "include/Armor.hpp"
#include "include/HikCamera.hpp"
#include "include/RTSerial.hpp"
#include "include/fastqueue.hpp"
#include "include/Yolo.hpp"
#include "include/Solver.hpp"
#include "include/Shooter.hpp"
#include "include/Target.hpp"
// #include "../../rm-main/include/Tracker.hpp"
#include "include/ShootTable.hpp"
#include "include/Data.hpp"
#include "Function.hpp"
#include "include/RerunVisualizer.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>

#define MainDebug
#ifdef MainDebug
double R_sum = 0.0;
int R_count = 0;
#endif
//Debug


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

io::HikCamera Hik(1.5,15);
io::RTSerial<Packet> ser(20);


YOLO11Detector yolo11detect("../model/yolo11.xml",YOLO11Detector::Camp::Blue);

RerunVisualizer viz("RoboMaster_AutoAim");

Solver Sov("../../config/Solver_config.yaml");
Robot robot;
// Tracker track;


ShootTable::TableConfig tableconfig(10,0,2,-1,0.01,"../../config/infantry_10_table.bin");
Shooter shoot(cv::Point3d(-0.9972026403208731,0.001749666619733665, -0.07472504803477144),tableconfig);

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

    cv::namedWindow("frame");
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
        //计算枪管方向
        const auto& Gun = shoot.GunDirection(frame.quat);


        auto armors = yolo11detect(frame.image);
        // yolo11detect.draw(frame.image,armors);
        // cv::imshow("frame",frame.image);
        // cv::waitKey(1);

        // std::cout<<"------------------------------------------------\n";

        // std::cout<<"detect num: "<<armors.size()<<"\n";

        //解算装甲板位置
        auto armors_posi = Sov(armors);

        // std::cout<<"after filter num: "<<armors_posis.size()<<"\n";

        // std::cout<<"after classify num: "<<armors_posi.size()<<"\n";

        // for(const auto& armor_posi : armors_posi)
        // {
        //     Sov.ansShow(armor_posi.posi,frame.image);
        // }

        // cv::imshow("frame", frame.image);
        
        // cv::waitKey(1);

        

        Sov.ConverToWorld(armors_posi,frame.quat);
        if(armors_posi.empty()) {robot.Update(rm::SolveDt(next_point, frame.time,0.005));}
        else(robot.Update(armors_posi,rm::SolveDt(next_point, frame.time,0.005)));
        next_point = frame.time;

        // #ifdef MainDebug
        // std::cout<<"------------------------------------------------\n";
        // std::cout<<"armors_posi: "<< "\n" <<armors_posi[0].posi<<"\n";
        // #endif


        double dt = shoot.FlyTime(cv::Point3d(robot.center.x()/100.0, robot.center.y()/100.0, robot.center.z()/100.0));
        auto aims = robot.Predic(dt);

        #ifdef MainDebug
        if(test.num%4 == 0 && test.num != 0)
            viz.update(robot, aims, dt,  Gun);
        #endif

        auto aim =  rm::ChooseBestAimArmor(aims, robot.Speed, Gun);
        
        // if(test.num%100 == 0 && test.num != 0)
        // {
        //     std::cout<<armors_posi[0].posi/10<<"\n";
        // }
        // std::cout<<"quat: "<<frame.quat.w<<" "<<frame.quat.x<<" "<<frame.quat.y<<" "<<frame.quat.z<<"\n";

        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {test.show();test.clear();}


        //打弹
        
        // std::cout<< "aim: " << aim << "\n";
        
        auto predict_posi = cv::Point3d(aim(0,0)/100.0, aim(1,0)/100.0, aim(2,0)/100.0);//单位换算到m
        // auto predict_posi =  armors_posi[0].posi/100.0;
        // std::cout<< "Predict Position: " << predict_posi << "\n";

        std::array<double, 2> Pitch_and_Yaw = shoot(predict_posi);
        // if(0.1 < std::abs(Pitch_and_Yaw[1])  || std::abs(Pitch_and_Yaw[1]) < 0.2 )
        // {
        //     std::cerr<<"aim error: "<< Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n"<< robot.Speed<<"\n";
            
        // }else {
        //     std::cout<<"aim ok: "<< Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n"<< robot.Speed<<"\n";
        // }

        // std::cout<<Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n";
        // rm::SendMessageToRobot(ser, Pitch_and_Yaw[0], Pitch_and_Yaw[1] , true);
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

