// ROS2
#include "rm_rune_tracker/rune_tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>

#include <cmath>
#include <opencv2/opencv.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace rm_auto_aim
{

RuneTrackerNode::RuneTrackerNode(const rclcpp::NodeOptions& options)
    : Node("rune_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting RuneTrackerNode!");

  // Declare parameters
  target_frame_ = this->declare_parameter("target_frame", "odom");
  std::string robot_type = this->declare_parameter("robot_type", "standard");

  // TF2 setup
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // Trajectory solver parameters
  float k = static_cast<float>(this->declare_parameter("tracker.k", 0.092));
  float bias_time =
      static_cast<float>(this->declare_parameter("tracker.bias_time", 0.01));
  float s_bias = static_cast<float>(this->declare_parameter("tracker.s_bias", 0.19133));
  float z_bias = static_cast<float>(this->declare_parameter("tracker.z_bias", 0.21265));
  float pitch_bias =
      static_cast<float>(this->declare_parameter("tracker.pitch_bias", 0.0));

  bool use_table = this->declare_parameter("tracker.calculate_mode", true);

  double max_x = this->declare_parameter("tracker.table.max_x", 13.0);
  double min_x = this->declare_parameter("tracker.table.min_x", 0.0);
  double max_y = this->declare_parameter("tracker.table.max_y", 2.0);
  double min_y = this->declare_parameter("tracker.table.min_y", -1.0);
  double resolution = this->declare_parameter("tracker.table.resolution", 0.01);
  std::string table_filename =
      this->declare_parameter("tracker.table.filename", "table.bin");

  SolveTrajectory::CalculateMode calculate_mode{};
  if (use_table)
  {
    calculate_mode = SolveTrajectory::CalculateMode::TABLE_LOOKUP;
  }
  else
  {
    calculate_mode = SolveTrajectory::CalculateMode::NORMAL;
  }

  TrajectoryTable::TableConfig table_config = {max_x, min_x,      max_y,
                                               min_y, resolution, table_filename};
  trajectory_solver_ = std::make_unique<SolveTrajectory>(
      k, bias_time, s_bias, z_bias, pitch_bias, calculate_mode, table_config);
  RCLCPP_INFO(this->get_logger(), "Trajectory solver initialized");

  // Subscribers
  rune_target_sub_ = this->create_subscription<rm_rune_interfaces::msg::RuneTarget>(
      "/rune/target", 10,
      std::bind(&RuneTrackerNode::RuneTargetCallback, this, std::placeholders::_1));

  velocity_sub_ = this->create_subscription<auto_aim_interfaces::msg::Velocity>(
      "/current_velocity", 10,
      std::bind(&RuneTrackerNode::VelocityCallback, this, std::placeholders::_1));

  debug_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/rune/debug_image", 10,
      std::bind(&RuneTrackerNode::DebugImageCallback, this, std::placeholders::_1));

  // Publishers
  send_pub_ = this->create_publisher<auto_aim_interfaces::msg::Send>("/tracker/send", 10);
  debug_image_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>("/rune_tracker/debug_image", 10);

  RCLCPP_INFO(this->get_logger(), "RuneTrackerNode initialized");
}

void RuneTrackerNode::RuneTargetCallback(
    const rm_rune_interfaces::msg::RuneTarget::SharedPtr msg)
{
  if (!msg->detected)
  {
    return;
  }

  // Filter by filter_valid to choose between raw and filtered data
  geometry_msgs::msg::Pose center_pose;
  double angular_velocity = NAN;
  if (msg->filter_valid)
  {
    center_pose = msg->filtered_center_pose;
    angular_velocity = msg->filtered_angular_velocity;
  }
  else
  {
    center_pose = msg->center_pose;
    angular_velocity = msg->angular_velocity;
  }

  // Find PENDING_STRUCK rune
  const rm_rune_interfaces::msg::Rune* target_rune = nullptr;
  for (const auto& rune : msg->runes)
  {
    if (rune.rune_type == rm_rune_interfaces::msg::Rune::PENDING_STRUCK)
    {
      target_rune = &rune;
      break;
    }
  }

  if (!target_rune)
  {
    RCLCPP_DEBUG(this->get_logger(), "No PENDING_STRUCK rune found");
    return;
  }

  // Transform target_rune->pose to target_frame_
  geometry_msgs::msg::PoseStamped pose_in_camera;
  pose_in_camera.header = msg->header;
  pose_in_camera.pose = target_rune->pose;

  geometry_msgs::msg::PoseStamped pose_in_target;
  try
  {
    pose_in_target = tf2_buffer_->transform(pose_in_camera, target_frame_);
  }
  catch (const tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(), "TF2 transform failed: %s", ex.what());
    return;
  }

  // Iterative trajectory solving
  float pitch = 0.0f, yaw = 0.0f;
  bool solved = false;
  constexpr int max_iterations = 20;

  for (int i = 0; i < max_iterations; ++i)
  {
    // Predict target position after fly_time
    double fly_time = 0.0;
    if (current_velocity_ > 1e-3)
    {
      double dx = pose_in_target.pose.position.x;
      double dy = pose_in_target.pose.position.y;
      double dz = pose_in_target.pose.position.z;
      double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      fly_time = dist / current_velocity_;
    }

    // Simple prediction: target rotates by angular_velocity * fly_time
    double predicted_angle_offset =
        target_rune->angle_offset + angular_velocity * fly_time;

    // Convert predicted position in polar coords
    // Assume rune center is transformed the same way
    geometry_msgs::msg::PoseStamped center_in_camera;
    center_in_camera.header = msg->header;
    center_in_camera.pose = center_pose;
    geometry_msgs::msg::PoseStamped center_in_target;
    try
    {
      center_in_target = tf2_buffer_->transform(center_in_camera, target_frame_);
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_ERROR(this->get_logger(), "Center TF2 transform failed: %s", ex.what());
      return;
    }

    // Compute radius in XY plane
    double rx = pose_in_target.pose.position.x - center_in_target.pose.position.x;
    double ry = pose_in_target.pose.position.y - center_in_target.pose.position.y;
    double radius = std::sqrt(rx * rx + ry * ry);

    // Predicted target position
    double predicted_x =
        center_in_target.pose.position.x + radius * std::cos(predicted_angle_offset);
    double predicted_y =
        center_in_target.pose.position.y + radius * std::sin(predicted_angle_offset);
    double predicted_z = pose_in_target.pose.position.z;

    // Call SolvePitch and SolveYaw
    float new_pitch = trajectory_solver_->SolvePitch(static_cast<float>(predicted_x),
                                                     static_cast<float>(predicted_y),
                                                     static_cast<float>(predicted_z));
    float new_yaw = trajectory_solver_->SolveYaw(static_cast<float>(predicted_x),
                                                 static_cast<float>(predicted_y));

    // Check for NaN from solver
    if (std::isnan(new_pitch) || std::isnan(new_yaw))
    {
      solved = false;
      RCLCPP_WARN(this->get_logger(), "SolvePitch/SolveYaw returned NaN at iteration %d",
                  i);
      break;
    }

    solved = true;

    if (!solved)
    {
      RCLCPP_WARN(this->get_logger(), "SolveTrajectory failed at iteration %d", i);
      break;
    }

    // Check convergence
    if (std::abs(new_pitch - pitch) < 0.001f && std::abs(new_yaw - yaw) < 0.001f)
    {
      pitch = new_pitch;
      yaw = new_yaw;
      break;
    }

    pitch = new_pitch;
    yaw = new_yaw;
  }

  // NaN protection
  if (std::isnan(pitch) || std::isnan(yaw))
  {
    RCLCPP_WARN(this->get_logger(), "pitch or yaw is NaN!");
    return;
  }

  // Publish Send message
  auto send_msg = auto_aim_interfaces::msg::Send();
  send_msg.header = msg->header;
  send_msg.pitch = pitch;
  send_msg.yaw = yaw;
  send_msg.is_fire = solved;
  send_msg.idx = 0;

  send_pub_->publish(send_msg);
}

void RuneTrackerNode::VelocityCallback(
    const auto_aim_interfaces::msg::Velocity::SharedPtr msg)
{
  current_velocity_ = msg->velocity;
  RCLCPP_DEBUG(this->get_logger(), "Updated bullet velocity: %.2f m/s",
               current_velocity_);
}

void RuneTrackerNode::DebugImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Convert to OpenCV Mat
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (const cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // Overlay debug text
  // TODO: Capture pitch/yaw/filter_valid from RuneTargetCallback and display here
  std::string text = "RuneTracker Debug";
  cv::putText(cv_ptr->image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 255, 0), 2);

  // Publish to /rune_tracker/debug_image
  debug_image_pub_->publish(*cv_ptr->toImageMsg());
}

}  // namespace rm_auto_aim

RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::RuneTrackerNode)
