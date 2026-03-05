#ifndef RM_RUNE_TRACKER__RUNE_TRACKER_NODE_HPP_
#define RM_RUNE_TRACKER__RUNE_TRACKER_NODE_HPP_

// ROS
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

// Interface
#include "armor_tracker/SolveTrajectory.hpp"
#include "auto_aim_interfaces/msg/send.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"
#include "rm_rune_interfaces/msg/rune_target.hpp"

namespace rm_auto_aim
{

class RuneTrackerNode : public rclcpp::Node
{
 public:
  explicit RuneTrackerNode(const rclcpp::NodeOptions& options);

 private:
  void RuneTargetCallback(const rm_rune_interfaces::msg::RuneTarget::SharedPtr msg);
  void VelocityCallback(const auto_aim_interfaces::msg::Velocity::SharedPtr msg);
  void DebugImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

  // TF2
  std::string target_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  // Subscribers
  rclcpp::Subscription<rm_rune_interfaces::msg::RuneTarget>::SharedPtr rune_target_sub_;
  rclcpp::Subscription<auto_aim_interfaces::msg::Velocity>::SharedPtr velocity_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr debug_image_sub_;

  // Publishers
  rclcpp::Publisher<auto_aim_interfaces::msg::Send>::SharedPtr send_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

  // Ballistic trajectory solver
  std::unique_ptr<SolveTrajectory> trajectory_solver_;

  // Current bullet velocity
  double current_velocity_{16.0};
};

}  // namespace rm_auto_aim

#endif  // RM_RUNE_TRACKER__RUNE_TRACKER_NODE_HPP_
