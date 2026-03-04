/**
 * @file rune_detector_node.hpp
 * @brief ROS2 能量机关检测节点
 * @details 对华南虎战队 rm_vision_core 能量机关算法的 ROS2 封装。
 *          订阅相机图像与相机参数，利用 tf2 获取云台位姿，
 *          输出能量机关中心三维位姿及各扇叶的估计位姿、速度、激活状态。
 *          支持 RViz 3D 可视化和调试图像发布。
 */

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rm_rune_interfaces/msg/rune_target.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// rm_vision_core 原始库头文件
#include <vc/camera/camera_param.h>
#include <vc/detector/rune_detector.h>
#include <vc/feature/rune_center.h>
#include <vc/feature/rune_combo.h>
#include <vc/feature/rune_fan.h>
#include <vc/feature/rune_group.h>
#include <vc/feature/rune_target.h>
#include <vc/feature/rune_tracker.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rm_rune_detector
{

class RuneDetectorNode : public rclcpp::Node
{
 public:
  explicit RuneDetectorNode(const rclcpp::NodeOptions& options);

 private:
  // ======================== 回调函数 ========================
  void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg);
  void CameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg);

  // ======================== 辅助函数 ========================
  bool QueryGimbalPose(const rclcpp::Time& stamp, GyroData& gyro_data);

  rm_rune_interfaces::msg::RuneTarget BuildRuneTargetMsg(
      const std_msgs::msg::Header& header, const std::shared_ptr<RuneGroup>& rune_group);

  static geometry_msgs::msg::Pose PoseNodeToRosPose(const PoseNode& pose_node);

  void DeclareParameters();
  void SyncParametersToCore();

  // ======================== RViz 可视化 ========================
  /// @brief 发布 RViz 3D Marker 可视化 (观测目标 + 估计目标)
  void PublishRvizMarkers(const std_msgs::msg::Header& header,
                          const std::shared_ptr<RuneGroup>& rune_group);

  /// @brief 创建单个位姿的立方体 Marker
  visualization_msgs::msg::Marker CreateCubeMarker(const std_msgs::msg::Header& header,
                                                   int id, const PoseNode& pose, float r,
                                                   float g, float b, float a, float sx,
                                                   float sy, float sz) const;

  /// @brief 创建坐标轴 Marker (箭头)
  visualization_msgs::msg::Marker CreateAxisMarker(const std_msgs::msg::Header& header,
                                                   int id, const PoseNode& pose,
                                                   int axis,  // 0=x, 1=y, 2=z
                                                   float length) const;

  /// @brief 创建文本 Marker
  visualization_msgs::msg::Marker CreateTextMarker(const std_msgs::msg::Header& header,
                                                   int id,
                                                   const geometry_msgs::msg::Pose& pose,
                                                   const std::string& text, float r,
                                                   float g, float b) const;

  // ======================== 调试图像 ========================
  /// @brief 绘制完整的调试图像
  void DrawDebugImage(const cv::Mat& frame, const std_msgs::msg::Header& header,
                      const std::shared_ptr<RuneGroup>& rune_group,
                      const DetectorOutput& output, double process_time_ms);

  /// @brief 在图像上绘制3D立方体投影
  static void DrawCube(cv::Mat& img, const PoseNode& p, float x_len, float y_len,
                       float z_len, const cv::Scalar& color, int thickness = 1);

  /// @brief 在图像上绘制3D坐标轴投影
  static void DrawAxes(cv::Mat& img, const PoseNode& p, float length, int thickness = 2);

  /// @brief 在图像上绘制预测击打点
  void DrawPredictedPoint(cv::Mat& img, const std::shared_ptr<RuneGroup>& rune_group);

  // ======================== 成员变量 ========================
  // -- 订阅 --
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;

  // -- 发布 --
  rclcpp::Publisher<rm_rune_interfaces::msg::RuneTarget>::SharedPtr rune_pub_;
  image_transport::Publisher debug_img_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // -- tf2 --
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // -- 坐标系名称 --
  std::string target_frame_;
  std::string source_frame_;
  std::string camera_frame_;

  // -- 检测器核心 --
  std::unique_ptr<RuneDetector> detector_;
  std::vector<FeatureNode_ptr> rune_groups_;

  // -- 相机参数 --
  bool camera_info_received_ = false;
  cv::Matx33f camera_matrix_;
  cv::Matx<float, 5, 1> dist_coeffs_;

  // -- 检测参数 --
  int detect_color_;
  int color_threshold_;

  // -- 调试 --
  bool debug_mode_;

  // -- 上一帧二值化图 (用于调试显示) --
  cv::Mat last_binary_;
};

}  // namespace rm_rune_detector