#include "armor_tracker/tracker_node.hpp"
#include <cmath>
#include <memory>
#include <algorithm>

namespace rm_auto_aim
{
// 1. 节点初始化与订阅发布
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions& options)
    : Node("armor_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode (PIXEL PI CONTROL MODE)!");
  
  // 保留空壳防报错
  tracker_ = std::make_unique<Tracker>(0.0, 0.0);
  solver_ = std::make_unique<SolveTrajectory>(0.0);
  
  armors_sub_ = this->create_subscription<auto_aim_interfaces::msg::Armors>(
      "/detector/armors", rclcpp::SensorDataQoS(),
      std::bind(&ArmorTrackerNode::ArmorsCallback, this, std::placeholders::_1));

  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS());
  send_pub_ = this->create_publisher<auto_aim_interfaces::msg::Send>(
      "/tracker/send", rclcpp::SensorDataQoS());

  joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      std::bind(&ArmorTrackerNode::JointStateCallback, this, std::placeholders::_1));
}

// 2. 接收底层云台姿态
void ArmorTrackerNode::JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  for (size_t i = 0; i < msg->name.size(); ++i) {
    if (msg->name[i] == "pitch_joint" || msg->name[i] == "gimbal_pitch_joint") 
      current_pitch_ = msg->position[i];
    if (msg->name[i] == "yaw_joint" || msg->name[i] == "gimbal_yaw_joint")   
      current_yaw_ = msg->position[i];
  }
}

// 3. 核心视觉闭环控制
void ArmorTrackerNode::ArmorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  auto_aim_interfaces::msg::Send send_msg;
  send_msg.is_fire = false;

  static float last_target_yaw = current_yaw_;
  static float last_target_pitch = current_pitch_;
  static bool is_init = false;

  // 👑 新增神器：静差粉碎器（积分池），专治对不准红点！
  static float i_yaw = 0.0f;
  static float i_pitch = 0.0f;

  if (!is_init) {
      last_target_yaw = current_yaw_;
      last_target_pitch = current_pitch_;
      is_init = true;
  }

  // 丢失目标时，清空积分，立刻停在原地
  if (armors_msg->armors.empty()) {
    send_msg.pitch = last_target_pitch;
    send_msg.yaw = last_target_yaw;
    send_pub_->publish(send_msg);
    i_yaw = 0.0f;
    i_pitch = 0.0f;
    return;
  }

  // 挑最靠近中心的装甲板
  auto armor = armors_msg->armors[0];
  for (const auto& a : armors_msg->armors) {
      if (a.distance_to_image_center < armor.distance_to_image_center) {
          armor = a;
      }
  }

  // 像素误差
  float yaw_err = armor.pose.position.x;   
  float pitch_err = armor.pose.position.y; 

  // 极小死区：进入红点范围（0.002弧度）就认为完美对准，清空力气
  if (std::abs(yaw_err) < 0.002f) { yaw_err = 0.0f; i_yaw = 0.0f; }
  if (std::abs(pitch_err) < 0.002f) { pitch_err = 0.0f; i_pitch = 0.0f; }

  // 👑 积分累加：只要没对准红点，就在后台疯狂攒力气！
  float ki = 0.02f;
  i_yaw += yaw_err * ki;
  i_pitch += pitch_err * ki;

  // 限制力气上限，防止它攒太多飞车
  i_yaw = std::clamp(i_yaw, -0.05f, 0.05f);
  i_pitch = std::clamp(i_pitch, -0.05f, 0.05f);

  float kp = 0.6f; 
  
  // 👑 目标 = 当前 + 比例(P)推力 + 积分(I)推力 (使用你验证过的加号！)
  float raw_target_yaw = current_yaw_ - kp * yaw_err - i_yaw;     
  float raw_target_pitch = current_pitch_ + kp * pitch_err + i_pitch;

  // 防震荡限幅
  float max_step = 0.05f;
  raw_target_yaw = std::clamp(raw_target_yaw, current_yaw_ - max_step, current_yaw_ + max_step);
  raw_target_pitch = std::clamp(raw_target_pitch, current_pitch_ - max_step, current_pitch_ + max_step);

  // 丝滑滤波
  float alpha = 0.5f; 
  last_target_yaw = alpha * raw_target_yaw + (1.0f - alpha) * last_target_yaw;
  last_target_pitch = alpha * raw_target_pitch + (1.0f - alpha) * last_target_pitch;

  send_msg.pitch = last_target_pitch;
  send_msg.yaw = last_target_yaw;
  send_pub_->publish(send_msg);

  // 日志打印 I 项，你可以肉眼看着它是怎么发力把云台推向红点的
  RCLCPP_INFO(this->get_logger(), "Err_Y: %.3f, I_Y: %.3f, Tar_Y: %.3f", yaw_err, i_yaw, last_target_yaw);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)