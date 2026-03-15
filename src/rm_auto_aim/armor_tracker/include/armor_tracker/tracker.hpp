#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include <algorithm>

namespace rm_auto_aim
{

class Tracker
{
 public:
  Tracker(double max_match_distance, double max_match_yaw_diff);

  using Armors = auto_aim_interfaces::msg::Armors;
  using Armor = auto_aim_interfaces::msg::Armor;

  void Init(const Armors::SharedPtr& armors_msg);
  
  // 核心 Update 函数声明
  void Update(const Armors::SharedPtr& armors_msg, float cur_yaw, float cur_pitch, float yaw_err, float pitch_err);

  float get_final_yaw() const { return target_yaw_; }
  float get_final_pitch() const { return target_pitch_; }

  enum State : uint8_t
  {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;

  Armor tracked_armor;

 private:
  // 控制状态成员变量
  float target_yaw_ = 0.0f;
  float target_pitch_ = 0.0f;
  bool is_initialized_ = false;

  // 控制参数
  float kp_ = 1.2f;
  float alpha_ = 0.8f;
  float max_step_ = 0.06f;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_