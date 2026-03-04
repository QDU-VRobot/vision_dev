#pragma once

#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "vc/camera/camera_param.h"

namespace rune_detector
{
inline void update_camera_param_from_camera_info(const sensor_msgs::msg::CameraInfo& info)
{
  // K is row-major 3x3
  camera_param.cameraMatrix =
      cv::Matx33f(static_cast<float>(info.k[0]), static_cast<float>(info.k[1]),
                  static_cast<float>(info.k[2]), static_cast<float>(info.k[3]),
                  static_cast<float>(info.k[4]), static_cast<float>(info.k[5]),
                  static_cast<float>(info.k[6]), static_cast<float>(info.k[7]),
                  static_cast<float>(info.k[8]));

  // distCoeff: use first 5 if available, else zeros
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, d4 = 0.f;
  if (info.d.size() > 0)
  {
    d0 = static_cast<float>(info.d[0]);
  }
  if (info.d.size() > 1)
  {
    d1 = static_cast<float>(info.d[1]);
  }
  if (info.d.size() > 2)
  {
    d2 = static_cast<float>(info.d[2]);
  }
  if (info.d.size() > 3)
  {
    d3 = static_cast<float>(info.d[3]);
  }
  if (info.d.size() > 4)
  {
    d4 = static_cast<float>(info.d[4]);
  }
  camera_param.distCoeff = cv::Matx<float, 5, 1>(d0, d1, d2, d3, d4);

  camera_param.image_width = static_cast<int>(info.width);
  camera_param.image_height = static_cast<int>(info.height);
}
}  // namespace rune_detector