#ifndef TARGET_HPP
#define TARGET_HPP

#include "Armor.hpp"
#include "EKFKalman.hpp"

#include <deque>
#include <eigen3/Eigen/Core>
#include <array>
#include <opencv2/core/types.hpp>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

struct RobotSize
{
//目标半径,0小半径,1大半径
    std::array<double,2> radius{20.0,30.0}; 

    RobotSize() = default;

    void operator () (double r_small, double r_big)
    {
        radius[0] = r_small;
        radius[1] = r_big;
    }
    
};

class Robot
{
public:
    Robot() = default;

    enum class ArmorView : bool
    {
        Visual = true,
        Invisual = false
    };

    enum class Type : int
    {hero  = 1, two  = 2,
     three = 3, four = 4, 
     guard = 5} type;   
    

    /*!
    看见两个装甲板，更新Robot的尺寸
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    /注意：如果只传如一个装甲板，则不会更新尺寸
    */
    static void SolveRobotSize(std::vector<ArmorPosi>& armors);

    /*!
    第一次看见Robot，初始化Robot的状态
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    */
    void Init(const std::vector<ArmorPosi>& armors);

    /*!
    完全丢失Robot，清空。
    /回到未初始化状态
    */
    void Clear();

    /*!
    新数据到来，更新Robot的状态
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    /装甲板数量大于2时只会使用前两个装甲板进行更新，装甲板为空不做任何操作
    */
    void Update(const std::vector<ArmorPosi>& armors, const cv::Quatd& gripper_to_world, double dt);
    void Update(double dt);

    void OneArmor(const ArmorPosi& armor, const cv::Quatd& gripper_to_world, double dt);
    void TwoArmor(const std::vector<ArmorPosi>& armors, const cv::Quatd& gripper_to_world, double dt);

    /*!
    @return 返回当前装甲板的朝向角
    */
    double SolveTheta(const ArmorPosi& armors) ;

    Eigen::Matrix<double,4,4> Predic(double dt) ;//mm/s, rad/s

    /*!
    @return 旋转点Point绕轴axis旋转angle角度后的新坐标
    */
    Eigen::Matrix<double, 3, 1> Rotate( const Eigen::Matrix<double, 3, 1>& center, double angle );

public:
//整车速度v_x,v_y,v_z,w
    Eigen::Matrix<double, 4, 1> Speed;

/* 4个装甲板位置：
    ID:   0        1          2          3

    0     x        x          x          x
    1     y        y          y          y
    2     z        z          z          z
    3   theta    theta      theta      theta
    4   radius   radius     radius     radius
*/

/*
    theta radius h 
*/
    Eigen::Matrix<double, 3, 4> Armors;
    double l_diff = 0, h_diff = 0;
    const double r = 24.0f;
    double d_theta_1 = CV_PI/2, d_theta_2 = CV_PI, d_theta_3 = -CV_PI/2;

    std::array<ArmorView, 4> View = {ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual};


    //中心转轴方向
    const Eigen::Matrix<double,3,1> axis{0,0,1};//单位向量
    
    //中心点坐标
    Eigen::Matrix<double,3,1> center{0,0,0};

    EKFKalman Kalman;

private:

    //记录是否初始化
    bool is_init = false;
    
    static std::array<RobotSize, 5> Size;//不同类型机器人信息
};



// class Target
// {
// public:
//     virtual Eigen::Matrix<double,8,1> Predict(const Eigen::Matrix<double,8,1>& X,double dt) = 0;
//     virtual std::vector< Eigen::Matrix<double,8,1> > EveryArmorState(const std::vector< Eigen::Matrix<double,8,1> >& X) = 0;
    
//     /*!
//     @return 返回当前装甲板的朝向角
//     */
//     virtual double SolveAngel(const ArmorPosi& armors) = 0;

//     //中心转轴方向
//     cv::Point3d axis = cv::Point3d{0,0,1};//单位向量
    
//     //中心点坐标
//     Eigen::Matrix<double,3,1> center{0,0,0};

//     //对象速度向量
//     Eigen::Matrix<double,4,1> speed{0,0,0,0};


// };
    // //坐标系的变化矩阵
    // cv::Matx<double, 3, 3> R{1,0,0,
    //                          0,1,0,
    //                          0,0,1};


//中心转轴
    // std::queue<cv::Point3d> axis_set{std::deque<cv::Point3d>{cv::Point3d(0,0,1)}};
    // cv::Point3d axis_sum{0,0,1};

#endif // TARGET_HPP