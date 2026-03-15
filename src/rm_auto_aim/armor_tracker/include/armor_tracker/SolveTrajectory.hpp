#pragma once

#include <cmath>
#include <iostream>

#include "auto_aim_interfaces/msg/target.hpp"
#include "rclcpp/rclcpp.hpp"

namespace rm_auto_aim
{

//=============================================================================
// 弹道解算主类 (激光直瞄简化版)
//=============================================================================
class SolveTrajectory
{
 public:
  // 用于存储目标装甲板的信息
  struct TargetPostion
  {
    float x;    // 装甲板在世界坐标系下的x
    float y;    // 装甲板在世界坐标系下的y
    float z;    // 装甲板在世界坐标系下的z
    float yaw;  // 装甲板坐标系相对于世界坐标系的yaw角
  };

  // 构造函数：只接收云台响应延迟
  explicit SolveTrajectory(const float& bias_time);

  // 核心预测函数
  void PredictArmorPosition(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                            float time_delay);

  // 纯几何解算
// 纯几何解算
  float SolvePitch(float x, float y, float z);
  float SolveYaw(float x, float y, float z);

  // 对外唯一接口
  // 这里使用 const SharedPtr (值传递)
  void AutoSolveTrajectory(float& pitch, float& yaw, bool& is_fire, float& aim_x,
                           float& aim_y, float& aim_z,
                           const auto_aim_interfaces::msg::Target::SharedPtr msg);

 private:
  // 系统参数
  float bias_time_;  // 云台响应偏置时间

  // 预测过程变量
  float pre_x_center_{0.0f};
  float pre_y_center_{0.0f};
  float pre_z_center_{0.0f};
  float pre_yaw_{0.0f};

  // 存储预测后的装甲板位置
  struct TargetPostion pre_position_[4];
};

}  // namespace rm_auto_aim