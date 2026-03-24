#ifndef LIGHT_AND_ARMOR_STRUCT
#define LIGHT_AND_ARMOR_STRUCT
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/core/types.hpp>
#include <vector>
struct Light{
    explicit Light(const cv::RotatedRect& rect)
    {
        cv::Point2f p[4];  //灯条四个点
        rect.points(p);  //rotatedrect的points函数可以获取四个点的坐标
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        
        top = (p[0] + p[1]) / 2;//灯条顶部和底部中心点
        bottom = (p[2] + p[3]) / 2;

        this->center = rect.center;
        this->width = cv::norm(p[1]-p[0]);
        this->length = cv::norm(top - bottom);

    }

    cv::Point2f top, bottom;
    cv::Point2f center;

    double length;
    double width;
    enum class Color { Red, Blue } color;
};



struct Armor{
    Armor(const Light& light1, const Light& light2):
        left(light1), right(light2)
    {
        if(light1.center.x>light2.center.x) std::swap(left, right);
        Lightcorners.resize(4);
        Lightcorners[0] = left.top;
        Lightcorners[1] = right.top;
        Lightcorners[2] = right.bottom;
        Lightcorners[3] = left.bottom;
    }
    Light left, right;
    std::vector<cv::Point2f> Lightcorners; //装甲板四个顶点
 
};

struct ArmorPosi{
    cv::Point3d posi;
    cv::Point3d face;
    cv::Point3d toward;
    /*
    SCS 是装甲板在相机坐标系下的球坐标,
    包含三个分量：r（距离）、theta（仰角）和phi（方位角）
    */
    cv::Point3d SCS; 

    double theta;//装甲板在相机坐标系下的偏航角(rad)
    double error;
    enum  class Radius : int {Short = 0, Long = 1, Unknow = 2 } radius{Radius::Unknow};

    enum class Type : int {base    = 0, hero     = 1, two   = 2,
                           three   = 3, four     = 4, guard = 5,
                           outpost = 6} type;
    float confidence = 0; 

    ArmorPosi(cv::Point3d posi, cv::Point3d face, cv::Point3d toward, double theta, double error):
              posi(posi), face(face), toward(toward), theta(theta), error(error)
              {
                double& x = this->posi.x, & y = this->posi.y, & z = this->posi.z;
                double xx = x*x, yy = y*y, zz = z*z;
                this->SCS = cv::Point3d(std::sqrt(xx + yy + zz),
                                        std::atan2(std::sqrt(xx + yy), z),
                                        std::atan2(y, x));
              }
    };
struct YoloArmor{ 
    cv::Rect box;
    float conf;
    int class_id;
    std::vector<cv::Point2f> keypoints;
};
#endif
