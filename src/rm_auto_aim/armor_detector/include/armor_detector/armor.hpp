#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

namespace rm_auto_aim
{
const int RED = 0;
const int BLUE = 1;

enum class ArmorType : uint8_t
{
  SMALL,
  INVALID
};
const std::string ARMOR_TYPE_STR[2] = {"small", "invalid"};


struct Light : public cv::RotatedRect
{ 
  Light() = default;

  explicit Light(const cv::RotatedRect& box)
      : cv::RotatedRect(box)   // 只初始化父类
  {
    cv::Point2f p[4];  
    box.points(p);

    std::sort(p, p + 4,
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    left  = (p[0] + p[1]) / 2;
    right = (p[2] + p[3]) / 2;

    length = cv::norm(left - right);
    width  = cv::norm(p[0] - p[1]);

    // 计算倾角（你原来的公式）
    tilt_angle = std::atan2(
        std::abs(left.y - right.y),
        std::abs(left.x - right.x));

    tilt_angle = static_cast<float>(tilt_angle / CV_PI * 180);
  }

  int color;
  cv::Point2f left, right;
  double length;
  double width;
  float tilt_angle;
};


struct Armor
{
  Armor() = default;
  Armor(const Light& l1, const Light& l2)
  {
    if (l1.center.y < l2.center.y)
    {
      top_light = l1, bottom_light = l2;
    }
    else
    {
      top_light = l2, bottom_light = l1;
    }
    center = (top_light.center + bottom_light.center) / 2;
  }

  // Light pairs part
  Light top_light, bottom_light;
  cv::Point2f center;
  ArmorType type;

  // Number part
  // cv::Mat number_img;
  // std::string number;
  // float confidence;
  // std::string classification_result;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_