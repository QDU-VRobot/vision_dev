// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <cmath>
#include <vector>

#include "armor_detector/detector.hpp"

namespace rm_auto_aim
{
Detector::Detector(const int& bin_thres, const int& color, const LightParams& l,
                   const ArmorParams& a)
    : binary_thres(bin_thres), detect_color(color), l(l), a(a)
{
}

std::vector<Armor> Detector::Detect(const cv::Mat& input)
{
  binary_img = PreprocessImage(input);
  lights_ = FindLights(input, binary_img);
  armors_ = MatchLights(lights_);

  // if (!armors_.empty())
  // {
  //   classifier->ExtractNumbers(input, armors_);
  //   classifier->Classify(armors_);
  // }

  return armors_;
}

cv::Mat Detector::PreprocessImage(const cv::Mat& rgb_img)  // 图像预处理
{
  cv::Mat gray_img;
  cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 5)); 
  cv::morphologyEx(binary_img, binary_img, cv::MORPH_CLOSE, kernel);

  return binary_img;
}

std::vector<Light> Detector::FindLights(const cv::Mat& rbg_img, const cv::Mat& binary_img)
{
  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::findContours(
      binary_img, contours, hierarchy, cv::RETR_EXTERNAL,
      cv::CHAIN_APPROX_SIMPLE); 

  vector<Light> lights;
  this->debug_lights.data.clear();

  for (const auto& contour : contours)
  {
    if (contour.size() < 5) 
    {
      continue;
    }

    auto r_rect = cv::minAreaRect(contour);
    
    double contour_area = cv::contourArea(contour);
    double rect_area = r_rect.size.area();
    if (rect_area < 5.0 || (contour_area / rect_area) < 0.75) 
    {
      continue; 
    }

    auto light = Light(r_rect);

    if (IsLight(light))
    {
      auto rect = light.boundingRect();
      if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols &&
          0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= rbg_img.rows)
      {
        int sum_r = 0, sum_b = 0;
        auto roi = rbg_img(rect); 
        for (int i = 0; i < roi.rows; i++)
        {
          for (int j = 0; j < roi.cols; j++)
          {
            if (cv::pointPolygonTest(contour, cv::Point2f(static_cast<float>(j + rect.x), static_cast<float>(i + rect.y)), false) >= 0)
            { 
              sum_r += roi.at<cv::Vec3b>(i, j)[0];
              sum_b += roi.at<cv::Vec3b>(i, j)[2];
            }
          }
        }
        light.color = sum_r > sum_b ? RED : BLUE;
        lights.emplace_back(light);
      }
    }
  }
  return lights;
}

bool Detector::IsLight(const Light& light)
{
  // The ratio of light (short side / long side) 通关宽高比判断是否是灯条
  double ratio = light.width / light.length;
  bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

  bool angle_ok = light.tilt_angle < l.max_angle;

  bool is_light = ratio_ok && angle_ok;

  // Fill in debug information
  auto_aim_interfaces::msg::DebugLight light_data;
  light_data.center_x = static_cast<int>(light.center.x);
  light_data.ratio = static_cast<float>(ratio);
  light_data.angle = light.tilt_angle;
  light_data.is_light = is_light;
  this->debug_lights.data.emplace_back(light_data);

  return is_light;
}

std::vector<Armor> Detector::MatchLights(const std::vector<Light>& lights)
{
  std::vector<Armor> armors;
  this->debug_armors.data.clear();

  // 必须至少提取到2个以上的发光块，才有可能配对
  if (lights.size() < 2)
  {
    return armors;
  }

  // 【恢复经典：两两组合遍历配对】
  for (size_t i = 0; i < lights.size(); ++i)
  {
    for (size_t j = i + 1; j < lights.size(); ++j)
    {
      Light light_1 = lights[i];
      Light light_2 = lights[j];

      // 区分上下层：图像坐标系 Y 轴向下，Y 越小越在上面
      if (light_1.center.y > light_2.center.y)
      {
        std::swap(light_1, light_2);
      }

      // 将这一对灯条送入几何条件校验
      auto type = IsArmor(light_1, light_2);
      
      // 如果校验通过，成功生成装甲板！
      if (type != ArmorType::INVALID)
      {
        auto armor = Armor(light_1, light_2);
        
        armor.top_light = light_1;
        armor.bottom_light = light_2;
        armor.type = type;
        
        armors.emplace_back(armor);
      }
    }
  }

  return armors;
}

// Check if there is another light in the boundingRect formed by the 2 lights
// 判断是否存在干扰灯条
bool Detector::ContainLight(const Light& light_1, const Light& light_2,
                            const std::vector<Light>& lights)
{
  // // 1. 创建装甲板：用两个灯条的顶端和底端点构建最小外接矩形
  // auto points =
  //     std::vector<cv::Point2f>{light_1.left, light_1.right, light_2.left,
  //     light_2.right};
  // auto bounding_rect = cv::boundingRect(points);  // 生成整数坐标的矩形

  // // 2. 遍历所有灯条进行检查
  // for (const auto& test_light : lights)
  // {
  //   // 跳过当前正在配对的两个灯条（通过中心点坐标比较）
  //   if (test_light.center == light_1.center || test_light.center == light_2.center)
  //   {
  //     continue;
  //   }

  //   // 3. 检查其他灯条的关键点是否在装甲板内
  //   if (bounding_rect.contains(test_light.left) ||     // 顶点在区域内
  //       bounding_rect.contains(test_light.right) ||  // 底点在区域内
  //       bounding_rect.contains(test_light.center))
  //   {               // 中心点在区域内
  //     return true;  // 发现干扰灯条立即返回
  //   }
  // }

  return false;  // 遍历完成未发现干扰灯条
}

ArmorType Detector::IsArmor(const Light& light_1, const Light& light_2)
{
  double light_length_ratio = light_1.length < light_2.length
                                  ? light_1.length / light_2.length
                                  : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

  // 计算上下灯条的中心距离
  double avg_light_length = (light_1.length + light_2.length) / 2;
  double center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  
  // 【修改3】恢复真正的距离判断布尔值
  bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                             center_distance <= a.max_small_center_distance);

  cv::Point2f diff = light_1.center - light_2.center;
  double angle = std::abs(std::atan(diff.x / (diff.y + 1e-6))) / CV_PI * 180;
  bool angle_ok = angle < a.max_angle;

  // 【修改4】严格根据三大几何条件判断
  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  ArmorType type{};
  if (is_armor)
  {
    type = ArmorType::SMALL; // 全部统一判定为小装甲板即可
  }
  else
  {
    type = ArmorType::INVALID;
  }

  auto_aim_interfaces::msg::DebugArmor armor_data;
  armor_data.type = ARMOR_TYPE_STR[static_cast<int>(type)];
  armor_data.center_x = static_cast<int>((light_1.center.x + light_2.center.x) / 2);
  armor_data.light_ratio = static_cast<float>(light_length_ratio);
  armor_data.center_distance = static_cast<float>(center_distance);
  armor_data.angle = static_cast<float>(angle);
  this->debug_armors.data.emplace_back(armor_data);

  return type;
}

// cv::Mat Detector::
//     GetAllNumbersImage()  //
//     将检测到的所有装甲板上的数字图像垂直拼接成一个单独的图像并返回
// {
//   if (armors_.empty())
//   {
//     return cv::Mat(cv::Size(20, 28), CV_8UC1);
//   }
//   else
//   {
//     std::vector<cv::Mat> number_imgs;
//     number_imgs.reserve(armors_.size());
//     for (auto& armor : armors_)
//     {
//       number_imgs.emplace_back(armor.number_img);
//     }
//     cv::Mat all_num_img;
//     cv::vconcat(number_imgs, all_num_img);
//     return all_num_img;
//   }
// }

void Detector::DrawResults(cv::Mat& img)
{
  // Draw Lights
  for (const auto& light : lights_)
  {
    cv::circle(img, light.left, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.right, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color =
        (light.color == RED) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
    cv::line(img, light.left, light.right, line_color, 1);
  }

  // Draw armors
  for (const auto& armor : armors_)
  {
    cv::line(img, armor.top_light.left, armor.bottom_light.right, cv::Scalar(0, 255, 0),
             2);
    cv::line(img, armor.bottom_light.left, armor.top_light.right, cv::Scalar(0, 255, 0),
             2);
  }

  // Show numbers and confidence
  // for (const auto& armor : armors_)
  // {
  //   cv::putText(img, armor.classfication_result, armor.left_light.top,
  //               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
  // }
}

}  // namespace rm_auto_aim