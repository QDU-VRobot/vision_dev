#ifndef ARMOR_TRACKER__TRACKER_NODE_HPP_
#define ARMOR_TRACKER__TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
// 【新增】：包含 JointState 消息头文件
#include <sensor_msgs/msg/joint_state.hpp>

#include "armor_tracker/SolveTrajectory.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/send.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "armor_tracker/tracker.hpp"

namespace rm_auto_aim
{

class ArmorTrackerNode : public rclcpp::Node
{
 public:
  explicit ArmorTrackerNode(const rclcpp::NodeOptions& options);

 private:
  void ArmorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_ptr);
  
  // 【新增】：声明接收云台角度的回调函数
  void JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

  std::unique_ptr<Tracker> tracker_;
  std::unique_ptr<SolveTrajectory> solver_;

  // 最普通的订阅者和发布者
  rclcpp::Subscription<auto_aim_interfaces::msg::Armors>::SharedPtr armors_sub_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Send>::SharedPtr send_pub_;

  // 【新增】：订阅云台姿态的订阅者和变量
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  float current_pitch_ = 0.0f;
  float current_yaw_ = 0.0f;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_TRACKER__TRACKER_NODE_HPP_