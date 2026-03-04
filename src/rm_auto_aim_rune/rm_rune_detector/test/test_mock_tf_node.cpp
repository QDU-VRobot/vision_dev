/**
 * @file test_mock_tf_node.cpp
 * @brief 模拟 TF 发布节点
 * @details 发布静态的 gimbal_odom -> pitch_link -> camera_optical_frame TF 变换，
 *          用于在没有实际机器人的情况下测试检测节点。
 *          模拟云台在指定 yaw/pitch 角度下的位姿。
 *
 * 用法: ros2 run rm_rune_detector test_mock_tf_node
 *       --ros-args -p yaw:=0.0 -p pitch:=0.0
 */

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class MockTFNode : public rclcpp::Node
{
 public:
  MockTFNode() : Node("mock_tf_node")
  {
    this->declare_parameter<double>("yaw", 0.0);
    this->declare_parameter<double>("pitch", 0.0);
    this->declare_parameter<double>("camera_xyz_x", 0.10);
    this->declare_parameter<double>("camera_xyz_y", 0.0);
    this->declare_parameter<double>("camera_xyz_z", 0.05);
    // 相机内参 (用于发布 camera_info)
    this->declare_parameter<int>("image_width", 1280);
    this->declare_parameter<int>("image_height", 1024);
    this->declare_parameter<double>("fx", 1280.0);
    this->declare_parameter<double>("fy", 1280.0);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 512.0);
    this->declare_parameter<bool>("publish_camera_info", true);
    this->declare_parameter<double>("rate", 100.0);

    // 静态变换发布器
    static_br_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // 动态变换发布器 (gimbal yaw/pitch)
    dynamic_br_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // 相机参数发布
    if (this->get_parameter("publish_camera_info").as_bool())
    {
      cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
          "/camera_info", rclcpp::SensorDataQoS());
    }

    // 发布静态变换: pitch_link -> camera_link -> camera_optical_frame
    PublishStaticTransforms();

    // 定时发布动态变换
    double rate = this->get_parameter("rate").as_double();
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
        std::bind(&MockTFNode::PublishDynamicTransforms, this));

    RCLCPP_INFO(get_logger(), "Mock TF Node started. Publishing at %.0f Hz", rate);
  }

 private:
  void PublishStaticTransforms()
  {
    double cam_x = this->get_parameter("camera_xyz_x").as_double();
    double cam_y = this->get_parameter("camera_xyz_y").as_double();
    double cam_z = this->get_parameter("camera_xyz_z").as_double();

    // pitch_link -> camera_link (fixed offset)
    geometry_msgs::msg::TransformStamped cam_tf;
    cam_tf.header.stamp = this->now();
    cam_tf.header.frame_id = "pitch_link";
    cam_tf.child_frame_id = "camera_link";
    cam_tf.transform.translation.x = cam_x;
    cam_tf.transform.translation.y = cam_y;
    cam_tf.transform.translation.z = cam_z;
    tf2::Quaternion q_identity;
    q_identity.setRPY(0, 0, 0);
    cam_tf.transform.rotation.x = q_identity.x();
    cam_tf.transform.rotation.y = q_identity.y();
    cam_tf.transform.rotation.z = q_identity.z();
    cam_tf.transform.rotation.w = q_identity.w();
    static_br_->sendTransform(cam_tf);

    // camera_link -> camera_optical_frame (rpy = -pi/2, 0, -pi/2)
    geometry_msgs::msg::TransformStamped opt_tf;
    opt_tf.header.stamp = this->now();
    opt_tf.header.frame_id = "camera_link";
    opt_tf.child_frame_id = "camera_optical_frame";
    opt_tf.transform.translation.x = 0;
    opt_tf.transform.translation.y = 0;
    opt_tf.transform.translation.z = 0;
    tf2::Quaternion q_optical;
    q_optical.setRPY(-M_PI / 2, 0, -M_PI / 2);
    opt_tf.transform.rotation.x = q_optical.x();
    opt_tf.transform.rotation.y = q_optical.y();
    opt_tf.transform.rotation.z = q_optical.z();
    opt_tf.transform.rotation.w = q_optical.w();
    static_br_->sendTransform(opt_tf);

    RCLCPP_INFO(get_logger(),
                "Published static TFs: pitch_link -> camera_link (%.3f,%.3f,%.3f), "
                "camera_link -> camera_optical_frame",
                cam_x, cam_y, cam_z);
  }

  void PublishDynamicTransforms()
  {
    auto now = this->now();
    double yaw_deg = this->get_parameter("yaw").as_double();
    double pitch_deg = this->get_parameter("pitch").as_double();
    double yaw = yaw_deg * M_PI / 180.0;
    double pitch = pitch_deg * M_PI / 180.0;

    // gimbal_odom -> yaw_link (rotate around Z)
    geometry_msgs::msg::TransformStamped yaw_tf;
    yaw_tf.header.stamp = now;
    yaw_tf.header.frame_id = "gimbal_odom";
    yaw_tf.child_frame_id = "yaw_link";
    tf2::Quaternion q_yaw;
    q_yaw.setRPY(0, 0, yaw);
    yaw_tf.transform.rotation.x = q_yaw.x();
    yaw_tf.transform.rotation.y = q_yaw.y();
    yaw_tf.transform.rotation.z = q_yaw.z();
    yaw_tf.transform.rotation.w = q_yaw.w();

    // yaw_link -> pitch_link (rotate around -Y, matching URDF axis)
    geometry_msgs::msg::TransformStamped pitch_tf;
    pitch_tf.header.stamp = now;
    pitch_tf.header.frame_id = "yaw_link";
    pitch_tf.child_frame_id = "pitch_link";
    // URDF axis = "0 -1 0", so rotation is around -Y
    // pitch_joint angle rotates around axis (0,-1,0):
    //   equivalent to rotating -pitch around Y
    tf2::Quaternion q_pitch;
    q_pitch.setRPY(0, -pitch, 0);  // pitch around -Y = -pitch around +Y
    pitch_tf.transform.rotation.x = q_pitch.x();
    pitch_tf.transform.rotation.y = q_pitch.y();
    pitch_tf.transform.rotation.z = q_pitch.z();
    pitch_tf.transform.rotation.w = q_pitch.w();

    dynamic_br_->sendTransform(yaw_tf);
    dynamic_br_->sendTransform(pitch_tf);

    // 发布 camera_info
    if (cam_info_pub_)
    {
      sensor_msgs::msg::CameraInfo info;
      info.header.stamp = now;
      info.header.frame_id = "camera_optical_frame";
      info.width = this->get_parameter("image_width").as_int();
      info.height = this->get_parameter("image_height").as_int();
      double fx = this->get_parameter("fx").as_double();
      double fy = this->get_parameter("fy").as_double();
      double cx = this->get_parameter("cx").as_double();
      double cy = this->get_parameter("cy").as_double();
      info.k = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
      info.d = {0, 0, 0, 0, 0};
      info.distortion_model = "plumb_bob";
      info.p = {fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0};
      info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      cam_info_pub_->publish(info);
    }
  }

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_br_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_br_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockTFNode>());
  rclcpp::shutdown();
  return 0;
}
