/**
 * @file rune_detector_node.cpp
 * @brief ROS2 能量机关检测节点实现
 */

#include "rm_rune_detector/rune_detector_node.hpp"

#include <vc/detector/rune_detector_param.h>
#include <vc/feature/rune_center_param.h>
#include <vc/feature/rune_fan_param.h>
#include <vc/feature/rune_group_param.h>
#include <vc/feature/rune_target_param.h>
#include <vc/feature/rune_tracker_param.h>

#include <Eigen/Geometry>
#include <chrono>

namespace rm_rune_detector
{

// ============================================================================
// 构造函数
// ============================================================================
RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions& options)
    : Node("rune_detector", options)
{
  RCLCPP_INFO(get_logger(), "Initializing Rune Detector Node...");

  // 1. 声明参数
  DeclareParameters();

  // 2. 从参数服务器获取参数值
  detect_color_ = static_cast<int>(this->get_parameter("detect_color").as_int());
  color_threshold_ = static_cast<int>(this->get_parameter("color_threshold").as_int());
  debug_mode_ = this->get_parameter("debug").as_bool();
  target_frame_ = this->get_parameter("target_frame").as_string();
  source_frame_ = this->get_parameter("source_frame").as_string();
  camera_frame_ = this->get_parameter("camera_frame").as_string();

  // 3. 同步参数到 rm_vision_core 全局变量
  SyncParametersToCore();

  // 4. 初始化检测器
  detector_ = RuneDetector::make_detector();

  // 5. 初始化 tf2
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // 6. 订阅
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::SensorDataQoS(),
      std::bind(&RuneDetectorNode::ImageCallback, this, std::placeholders::_1));

  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_info", rclcpp::SensorDataQoS(),
      std::bind(&RuneDetectorNode::CameraInfoCallback, this, std::placeholders::_1));

  // 7. 发布
  rune_pub_ = this->create_publisher<rm_rune_interfaces::msg::RuneTarget>(
      "/rune/target", rclcpp::SensorDataQoS());

  // RViz MarkerArray 发布
  marker_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("/rune/markers", 10);

  // 调试图像发布 (总是创建，仅在 debug_mode 且有订阅者时才发送)
  debug_img_pub_ = image_transport::create_publisher(this, "/rune/debug_image");

  RCLCPP_INFO(get_logger(),
              "Rune Detector Node initialized. detect_color=%d, threshold=%d, debug=%s",
              detect_color_, color_threshold_, debug_mode_ ? "true" : "false");
  RCLCPP_INFO(get_logger(), "Frames: target=%s, source=%s, camera=%s",
              target_frame_.c_str(), source_frame_.c_str(), camera_frame_.c_str());
}

// ============================================================================
// 相机参数回调
// ============================================================================
void RuneDetectorNode::CameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg)
{
  camera_matrix_ =
      cv::Matx33f(static_cast<float>(info_msg->k[0]), static_cast<float>(info_msg->k[1]),
                  static_cast<float>(info_msg->k[2]), static_cast<float>(info_msg->k[3]),
                  static_cast<float>(info_msg->k[4]), static_cast<float>(info_msg->k[5]),
                  static_cast<float>(info_msg->k[6]), static_cast<float>(info_msg->k[7]),
                  static_cast<float>(info_msg->k[8]));

  if (info_msg->d.size() >= 5)
  {
    dist_coeffs_ = cv::Matx<float, 5, 1>(
        static_cast<float>(info_msg->d[0]), static_cast<float>(info_msg->d[1]),
        static_cast<float>(info_msg->d[2]), static_cast<float>(info_msg->d[3]),
        static_cast<float>(info_msg->d[4]));
  }
  else
  {
    dist_coeffs_ = cv::Matx<float, 5, 1>(0, 0, 0, 0, 0);
  }

  camera_param.cameraMatrix = camera_matrix_;
  camera_param.distCoeff = dist_coeffs_;
  camera_param.image_width = static_cast<int>(info_msg->width);
  camera_param.image_height = static_cast<int>(info_msg->height);

  if (!camera_info_received_)
  {
    camera_info_received_ = true;
    RCLCPP_INFO(get_logger(), "Camera info received: %dx%d, fx=%.1f, fy=%.1f",
                camera_param.image_width, camera_param.image_height, camera_matrix_(0, 0),
                camera_matrix_(1, 1));
  }
}

// ============================================================================
// 图像回调 —— 主检测流程
// ============================================================================
void RuneDetectorNode::ImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg)
{
  if (!camera_info_received_)
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info...");
    return;
  }

  auto t_start = std::chrono::steady_clock::now();

  // 1. 转换图像
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, "bgr8");
  }
  catch (const cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  const cv::Mat& frame = cv_ptr->image;
  if (frame.empty())
  {
    return;
  }

  // 2. 获取云台位姿
  GyroData gyro_data;
  bool has_tf = QueryGimbalPose(img_msg->header.stamp, gyro_data);
  if (!has_tf)
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Failed to query gimbal pose from tf2, using zero pose");
  }

  // 3. tick
  int64_t tick = static_cast<int64_t>(rclcpp::Time(img_msg->header.stamp).nanoseconds());

  // 4. 构造 DetectorInput
  DetectorInput input;
  input.setImage(frame);
  input.setGyroData(gyro_data);
  input.setTick(tick);
  input.setColor(detect_color_ == 0 ? PixChannel::RED : PixChannel::BLUE);
  input.setColorThresh(color_threshold_);
  input.setFeatureNodes(rune_groups_);

  // 5. 执行检测
  DetectorOutput output;
  try
  {
    detector_->detect(input, output);
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(get_logger(), "Detection exception: %s", e.what());
    rune_groups_.clear();
    return;
  }

  rune_groups_ = output.getFeatureNodes();

  auto t_end = std::chrono::steady_clock::now();
  double process_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  // 6. 构建并发布结果消息
  auto result_msg = rm_rune_interfaces::msg::RuneTarget();
  result_msg.header = img_msg->header;
  result_msg.header.frame_id = camera_frame_;

  std::shared_ptr<RuneGroup> rune_group = nullptr;
  if (output.getValid() && !rune_groups_.empty())
  {
    rune_group = RuneGroup::cast(rune_groups_.front());
  }

  if (!rune_group || rune_group->childFeatures().empty())
  {
    result_msg.detected = false;
    rune_pub_->publish(result_msg);
  }
  else
  {
    result_msg = BuildRuneTargetMsg(img_msg->header, rune_group);
    rune_pub_->publish(result_msg);

    // 7. RViz 可视化
    if (marker_pub_->get_subscription_count() > 0)
    {
      PublishRvizMarkers(img_msg->header, rune_group);
    }
  }

  // 8. 调试图像发布
  if (debug_mode_ && debug_img_pub_.getNumSubscribers() > 0)
  {
    DrawDebugImage(frame, img_msg->header, rune_group, output, process_ms);
  }
}

// ============================================================================
// 从 tf2 查询云台位姿
// ============================================================================
bool RuneDetectorNode::QueryGimbalPose(const rclcpp::Time& stamp, GyroData& gyro_data)
{
  try
  {
    // 查询 target_frame -> source_frame (gimbal_odom -> pitch_link)
    auto transform = tf_buffer_->lookupTransform(target_frame_, source_frame_, stamp,
                                                 rclcpp::Duration::from_seconds(0.02));

    const auto& q = transform.transform.rotation;
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 mat(tf_q);

    double yaw{}, pitch{}, roll{};
    mat.getEulerYPR(yaw, pitch, roll);

    // rm_vision_core 约定: yaw 向右为正, pitch 向下为正
    gyro_data.rotation.yaw = static_cast<float>(yaw * 180.0 / M_PI);
    gyro_data.rotation.pitch = static_cast<float>(pitch * 180.0 / M_PI);
    gyro_data.rotation.roll = static_cast<float>(roll * 180.0 / M_PI);
    gyro_data.rotation.yaw_speed = 0.0f;
    gyro_data.rotation.pitch_speed = 0.0f;
    gyro_data.rotation.roll_speed = 0.0f;

    // 查询 source_frame -> camera_frame (pitch_link -> camera_optical_frame)
    auto cam_to_gimbal = tf_buffer_->lookupTransform(
        source_frame_, camera_frame_, stamp, rclcpp::Duration::from_seconds(0.02));

    const auto& cq = cam_to_gimbal.transform.rotation;
    const auto& ct = cam_to_gimbal.transform.translation;
    tf2::Quaternion cam_q(cq.x, cq.y, cq.z, cq.w);
    tf2::Matrix3x3 cam_mat(cam_q);

    camera_param.cam2joint_rmat =
        cv::Matx33f(static_cast<float>(cam_mat[0][0]), static_cast<float>(cam_mat[0][1]),
                    static_cast<float>(cam_mat[0][2]), static_cast<float>(cam_mat[1][0]),
                    static_cast<float>(cam_mat[1][1]), static_cast<float>(cam_mat[1][2]),
                    static_cast<float>(cam_mat[2][0]), static_cast<float>(cam_mat[2][1]),
                    static_cast<float>(cam_mat[2][2]));

    // ROS 使用米, rm_vision_core 使用毫米
    camera_param.cam2joint_tvec = cv::Matx<float, 3, 1>(
        static_cast<float>(ct.x * 1000.0), static_cast<float>(ct.y * 1000.0),
        static_cast<float>(ct.z * 1000.0));

    return true;
  }
  catch (const tf2::TransformException& ex)
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF2 exception: %s",
                         ex.what());
    return false;
  }
}

// ============================================================================
// 构建 RuneTarget 消息
// ============================================================================
rm_rune_interfaces::msg::RuneTarget RuneDetectorNode::BuildRuneTargetMsg(
    const std_msgs::msg::Header& header, const std::shared_ptr<RuneGroup>& rune_group)
{
  rm_rune_interfaces::msg::RuneTarget msg;
  msg.header = header;
  msg.header.frame_id = camera_frame_;
  msg.detected = true;

  // 能量机关中心位姿
  auto& pose_cache = rune_group->getPoseCache();
  if (pose_cache.getPoseNodes().count(CoordFrame::CAMERA))
  {
    msg.center_pose = PoseNodeToRosPose(pose_cache.getPoseNodes().at(CoordFrame::CAMERA));
  }

  // 当前转角
  float current_angle = 0.0f;
  rune_group->getCurrentRotateAngle(current_angle);
  msg.current_angle = static_cast<double>(current_angle) * M_PI / 180.0;

  // 角速度估计
  const auto& raw_datas = rune_group->getRawDatas();
  const auto& history_ticks = rune_group->getHistoryTicks();
  if (raw_datas.size() >= 2 && history_ticks.size() >= 2)
  {
    double d_angle = raw_datas[0] - raw_datas[1];
    double d_time = static_cast<double>(history_ticks[0] - history_ticks[1]) * 1e-9;
    if (std::abs(d_time) > 1e-9)
    {
      msg.angular_velocity = (d_angle * M_PI / 180.0) / d_time;
    }
  }

  // 滤波数据处理
  PoseNode filtered_pose;
  msg.filter_valid = rune_group->getCamPnpDataFromFilter(filtered_pose);

  if (msg.filter_valid)
  {
    // 使用滤波位姿
    msg.filtered_center_pose = PoseNodeToRosPose(filtered_pose);

    // 计算滤波角速度
    if (raw_datas.size() >= 2 && history_ticks.size() >= 2)
    {
      float current_filtered_angle = 0.0f;
      rune_group->getCurrentRotateAngle(current_filtered_angle);

      // 使用滤波角度计算差分
      double d_angle = static_cast<double>(current_filtered_angle) - raw_datas[1];
      double d_time = static_cast<double>(history_ticks[0] - history_ticks[1]) * 1e-9;
      if (std::abs(d_time) > 1e-9)
      {
        msg.filtered_angular_velocity = (d_angle * M_PI / 180.0) / d_time;
      }
    }
  }
  else
  {
    // 滤波无效，回退为原始数据
    msg.filtered_center_pose = msg.center_pose;
    msg.filtered_angular_velocity = msg.angular_velocity;
  }

  // 各扇叶信息
  auto trackers = rune_group->getTrackers();
  for (const auto& tracker_node : trackers)
  {
    auto tracker = RuneTracker::cast(tracker_node);
    if (!tracker || tracker->getHistoryNodes().empty())
    {
      continue;
    }
    auto combo = RuneCombo::cast(tracker->getHistoryNodes().front());
    if (!combo)
    {
      continue;
    }

    rm_rune_interfaces::msg::Rune rune_msg;

    switch (combo->getRuneType())
    {
      case RuneType::STRUCK:
        rune_msg.rune_type = rm_rune_interfaces::msg::Rune::STRUCK;
        break;
      case RuneType::UNSTRUCK:
        rune_msg.rune_type = rm_rune_interfaces::msg::Rune::UNSTRUCK;
        break;
      case RuneType::PENDING_STRUCK:
        rune_msg.rune_type = rm_rune_interfaces::msg::Rune::PENDING_STRUCK;
        break;
      default:
        rune_msg.rune_type = rm_rune_interfaces::msg::Rune::UNKNOWN;
        break;
    }

    if (combo->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
    {
      rune_msg.pose =
          PoseNodeToRosPose(combo->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA));
    }

    // 速度差分
    if (tracker->getHistoryNodes().size() >= 2 && tracker->getHistoryTicks().size() >= 2)
    {
      auto prev_combo = RuneCombo::cast(tracker->getHistoryNodes()[1]);
      if (prev_combo &&
          prev_combo->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA) &&
          combo->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
      {
        const auto& t_now =
            combo->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA).tvec();
        const auto& t_prev =
            prev_combo->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA).tvec();
        double dt = static_cast<double>(tracker->getHistoryTicks()[0] -
                                        tracker->getHistoryTicks()[1]) *
                    1e-9;
        if (std::abs(dt) > 1e-9)
        {
          rune_msg.velocity.x = (t_now(0) - t_prev(0)) / dt * 0.001;
          rune_msg.velocity.y = (t_now(1) - t_prev(1)) / dt * 0.001;
          rune_msg.velocity.z = (t_now(2) - t_prev(2)) / dt * 0.001;
        }
      }
    }

    rune_msg.drop_frame_count = tracker->getDropFrameCount();
    msg.runes.push_back(rune_msg);
  }

  return msg;
}

// ============================================================================
// PoseNode -> geometry_msgs::Pose
// ============================================================================
geometry_msgs::msg::Pose RuneDetectorNode::PoseNodeToRosPose(const PoseNode& pose_node)
{
  geometry_msgs::msg::Pose pose;

  // mm -> m
  pose.position.x = pose_node.tvec()(0) * 0.001;
  pose.position.y = pose_node.tvec()(1) * 0.001;
  pose.position.z = pose_node.tvec()(2) * 0.001;

  // rvec -> 旋转矩阵 -> 四元数
  const auto& rmat = pose_node.rmat();
  Eigen::Matrix3d eigen_rmat;
  eigen_rmat << rmat(0, 0), rmat(0, 1), rmat(0, 2), rmat(1, 0), rmat(1, 1), rmat(1, 2),
      rmat(2, 0), rmat(2, 1), rmat(2, 2);
  Eigen::Quaterniond q(eigen_rmat);
  q.normalize();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();

  return pose;
}

// ============================================================================
// RViz 3D 可视化
// ============================================================================
void RuneDetectorNode::PublishRvizMarkers(const std_msgs::msg::Header& header,
                                          const std::shared_ptr<RuneGroup>& rune_group)
{
  visualization_msgs::msg::MarkerArray marker_array;
  int id = 0;

  auto marker_header = header;
  marker_header.frame_id = camera_frame_;

  // --- 1. 能量机关中心: 估计位姿 (绿色大球) ---
  if (rune_group->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
  {
    const auto& center_pose =
        rune_group->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);

    // 绿色半透明球体
    auto center_marker = CreateCubeMarker(marker_header, id++, center_pose, 0.0f, 1.0f,
                                          0.0f, 0.7f, 0.1f, 0.1f, 0.1f);
    center_marker.type = visualization_msgs::msg::Marker::SPHERE;
    center_marker.scale.x = 0.15;
    center_marker.scale.y = 0.15;
    center_marker.scale.z = 0.15;
    marker_array.markers.push_back(center_marker);

    // 坐标轴
    for (int axis = 0; axis < 3; ++axis)
    {
      marker_array.markers.push_back(
          CreateAxisMarker(marker_header, id++, center_pose, axis, 0.3f));
    }

    // 中心文本
    auto text_pose = PoseNodeToRosPose(center_pose);
    text_pose.position.z += 0.2;
    double dist_mm = cv::norm(center_pose.tvec());
    std::string text = "Center d=" + std::to_string(static_cast<int>(dist_mm)) + "mm";
    marker_array.markers.push_back(
        CreateTextMarker(marker_header, id++, text_pose, text, 1.0f, 1.0f, 0.0f));
  }

  // --- 2. 各追踪器 (扇叶): 根据类型着色的立方体 ---
  auto trackers = rune_group->getTrackers();
  for (const auto& tracker_node : trackers)
  {
    auto tracker = RuneTracker::cast(tracker_node);
    if (!tracker || tracker->getHistoryNodes().empty())
    {
      continue;
    }

    // 观测位姿 (tracker 自身的 pose cache)
    if (tracker->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
    {
      const auto& pose = tracker->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);

      auto combo = RuneCombo::cast(tracker->getHistoryNodes().front());
      RuneType type = combo ? combo->getRuneType() : RuneType::UNKNOWN;

      float r = 0.7f, g = 0.7f, b = 0.7f;
      std::string type_str = "UNK";
      if (type == RuneType::STRUCK)
      {
        r = 0.0f;
        g = 1.0f;
        b = 0.0f;
        type_str = "STRUCK";
      }
      else if (type == RuneType::PENDING_STRUCK)
      {
        r = 1.0f;
        g = 1.0f;
        b = 0.0f;
        type_str = "PENDING";
      }
      else if (type == RuneType::UNSTRUCK)
      {
        r = 1.0f;
        g = 1.0f;
        b = 1.0f;
        type_str = "UNSTRUCK";
      }

      // 估计位姿立方体 (500x500x300 mm = 0.5x0.5x0.3 m)
      auto cube =
          CreateCubeMarker(marker_header, id++, pose, r, g, b, 0.5f, 0.5f, 0.5f, 0.3f);
      marker_array.markers.push_back(cube);

      // 坐标轴
      for (int axis = 0; axis < 3; ++axis)
      {
        marker_array.markers.push_back(
            CreateAxisMarker(marker_header, id++, pose, axis, 0.2f));
      }

      // 类型标签
      auto text_pose = PoseNodeToRosPose(pose);
      text_pose.position.z += 0.15;
      marker_array.markers.push_back(
          CreateTextMarker(marker_header, id++, text_pose, type_str, r, g, b));
    }
  }

  // --- 3. 清理残留 Marker ---
  // 发布 DELETE_ALL + 新数据
  visualization_msgs::msg::MarkerArray delete_array;
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_marker.header = marker_header;
  delete_array.markers.push_back(delete_marker);
  marker_pub_->publish(delete_array);

  marker_pub_->publish(marker_array);
}

visualization_msgs::msg::Marker RuneDetectorNode::CreateCubeMarker(
    const std_msgs::msg::Header& header, int id, const PoseNode& pose, float r, float g,
    float b, float a, float sx, float sy, float sz) const
{
  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = "rune_detector";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::CUBE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose = PoseNodeToRosPose(pose);
  m.scale.x = sx;
  m.scale.y = sy;
  m.scale.z = sz;
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;
  m.color.a = a;
  m.lifetime = rclcpp::Duration::from_seconds(0.2);
  return m;
}

visualization_msgs::msg::Marker RuneDetectorNode::CreateAxisMarker(
    const std_msgs::msg::Header& header, int id, const PoseNode& pose, int axis,
    float length) const
{
  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = "rune_axes";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::ARROW;
  m.action = visualization_msgs::msg::Marker::ADD;

  auto ros_pose = PoseNodeToRosPose(pose);
  m.pose = ros_pose;

  geometry_msgs::msg::Point start, end;
  start.x = start.y = start.z = 0.0;
  end.x = (axis == 0) ? length : 0.0;
  end.y = (axis == 1) ? length : 0.0;
  end.z = (axis == 2) ? length : 0.0;
  m.points.push_back(start);
  m.points.push_back(end);

  m.scale.x = 0.01;  // shaft
  m.scale.y = 0.02;  // head
  m.color.r = (axis == 0) ? 1.0f : 0.0f;
  m.color.g = (axis == 1) ? 1.0f : 0.0f;
  m.color.b = (axis == 2) ? 1.0f : 0.0f;
  m.color.a = 1.0f;
  m.lifetime = rclcpp::Duration::from_seconds(0.2);
  return m;
}

visualization_msgs::msg::Marker RuneDetectorNode::CreateTextMarker(
    const std_msgs::msg::Header& header, int id, const geometry_msgs::msg::Pose& pose,
    const std::string& text, float r, float g, float b) const
{
  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = "rune_text";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose = pose;
  m.scale.z = 0.06;
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;
  m.color.a = 1.0f;
  m.text = text;
  m.lifetime = rclcpp::Duration::from_seconds(0.2);
  return m;
}

// ============================================================================
// 调试图像绘制
// ============================================================================
void RuneDetectorNode::DrawDebugImage(const cv::Mat& frame,
                                      const std_msgs::msg::Header& header,
                                      const std::shared_ptr<RuneGroup>& rune_group,
                                      [[maybe_unused]] const DetectorOutput& output,
                                      double process_time_ms)
{
  cv::Mat debug_img = frame.clone();

  // --- 1. 原始特征绘制 (调用 rm_vision_core 自带的 drawFeature) ---
  if (rune_group)
  {
    rune_group->drawFeature(debug_img);
  }

  // --- 2. 附加信息: 3D坐标轴投影、预测点 ---
  if (rune_group && rune_group->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
  {
    const auto& center_pose =
        rune_group->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);

    // 绘制中心位姿的3D坐标轴
    DrawAxes(debug_img, center_pose, 300.0f, 2);

    // 中心点十字标记
    std::vector<cv::Point2f> center_proj;
    std::vector<cv::Point3f> center_3d = {{0, 0, 0}};
    cv::projectPoints(center_3d, center_pose.rvec(), center_pose.tvec(),
                      camera_param.cameraMatrix, camera_param.distCoeff, center_proj);
    if (!center_proj.empty())
    {
      cv::drawMarker(debug_img, center_proj[0], cv::Scalar(0, 255, 0), cv::MARKER_CROSS,
                     20, 2);
    }

    // 绘制预测击打点
    DrawPredictedPoint(debug_img, rune_group);
  }

  // --- 3. 各扇叶 3D 立方体 + 类型标签 ---
  if (rune_group)
  {
    auto trackers = rune_group->getTrackers();
    for (const auto& tracker_node : trackers)
    {
      auto tracker = RuneTracker::cast(tracker_node);
      if (!tracker || tracker->getHistoryNodes().empty())
      {
        continue;
      }

      auto combo = RuneCombo::cast(tracker->getHistoryNodes().front());
      if (!combo)
      {
        continue;
      }

      RuneType type = combo->getRuneType();
      cv::Scalar color;
      std::string label;
      if (type == RuneType::STRUCK)
      {
        color = cv::Scalar(0, 255, 0);
        label = "STRUCK";
      }
      else if (type == RuneType::PENDING_STRUCK)
      {
        color = cv::Scalar(0, 255, 255);
        label = "PENDING";
      }
      else if (type == RuneType::UNSTRUCK)
      {
        color = cv::Scalar(255, 255, 255);
        label = "UNSTRUCK";
      }
      else
      {
        color = cv::Scalar(128, 128, 128);
        label = "UNK";
      }

      // 绘制 combo 自身的特征 (靶心、扇叶、中心)
      if (combo->getChildFeatures().count("target"))
      {
        auto target = combo->getChildFeatures().at("target");
        if (target)
        {
          target->drawFeature(debug_img);
        }
      }
      if (combo->getChildFeatures().count("center"))
      {
        auto center = combo->getChildFeatures().at("center");
        if (center)
        {
          center->drawFeature(debug_img);
        }
        if (combo->getChildFeatures().count("fan"))
        {
          auto fan = combo->getChildFeatures().at("fan");
          if (fan)
          {
            fan->drawFeature(debug_img);
          }
        }
      }

      // 如果 tracker 有自己的 pose, 绘制 3D 立方体
      if (tracker->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
      {
        const auto& tpose = tracker->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);
        DrawCube(debug_img, tpose, 500, 500, 300, color, 2);
        DrawAxes(debug_img, tpose, 200.0f, 1);

        // 在投影中心绘制标签
        std::vector<cv::Point2f> proj;
        std::vector<cv::Point3f> pts3d = {{0, 0, 0}};
        cv::projectPoints(pts3d, tpose.rvec(), tpose.tvec(), camera_param.cameraMatrix,
                          camera_param.distCoeff, proj);
        if (!proj.empty())
        {
          cv::putText(debug_img, label, proj[0] + cv::Point2f(5, -10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        }
      }
    }
  }

  // --- 4. 信息覆盖层 ---
  int y = 25;
  auto put_info =
      [&](const std::string& text, const cv::Scalar& c = cv::Scalar(0, 255, 0))
  {
    cv::putText(debug_img, text, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.55, c, 1,
                cv::LINE_AA);
    y += 22;
  };

  put_info("Rune Detector [" + std::string(detect_color_ == 0 ? "RED" : "BLUE") + "]",
           cv::Scalar(0, 200, 255));
  put_info("Process: " + std::to_string(static_cast<int>(process_time_ms)) + " ms");
  [[maybe_unused]] int ret{};
  if (rune_group)
  {
    put_info("Detected: YES", cv::Scalar(0, 255, 0));

    float angle = 0;
    rune_group->getCurrentRotateAngle(angle);
    char buf[128];
    ret = snprintf(buf, sizeof(buf), "Angle: %.1f deg", angle);
    put_info(buf);

    const auto& raw = rune_group->getRawDatas();
    const auto& ticks = rune_group->getHistoryTicks();
    if (raw.size() >= 2 && ticks.size() >= 2)
    {
      double dt = static_cast<double>(ticks[0] - ticks[1]) * 1e-9;
      if (std::abs(dt) > 1e-9)
      {
        double omega = (raw[0] - raw[1]) / dt;
        ret = snprintf(buf, sizeof(buf), "AngVel: %.1f deg/s", omega);
        put_info(buf);
      }
    }

    if (rune_group->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
    {
      double dist = cv::norm(
          rune_group->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA).tvec());
      ret = snprintf(buf, sizeof(buf), "Dist: %.0f mm", dist);
      put_info(buf);
    }

    // 追踪器概况
    int n_trackers = 0;
    auto trackers = rune_group->getTrackers();
    for (auto& t : trackers)
    {
      auto tk = RuneTracker::cast(t);
      if (tk && !tk->getHistoryNodes().empty())
      {
        n_trackers++;
      }
    }
    ret = snprintf(buf, sizeof(buf), "Trackers: %d/5", n_trackers);
    put_info(buf);
  }
  else
  {
    put_info("Detected: NO", cv::Scalar(0, 0, 255));
  }

  // 发布调试图像
  auto debug_msg = cv_bridge::CvImage(header, "bgr8", debug_img).toImageMsg();
  debug_img_pub_.publish(debug_msg);
}

// ============================================================================
// 3D 绘制辅助函数
// ============================================================================
void RuneDetectorNode::DrawCube(cv::Mat& img, const PoseNode& p, float x_len, float y_len,
                                float z_len, const cv::Scalar& color, int thickness)
{
  float hx = x_len / 2, hy = y_len / 2, hz = z_len / 2;
  std::vector<cv::Point3f> pts3d = {{-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz},
                                    {-hx, hy, -hz},  {-hx, -hy, hz}, {hx, -hy, hz},
                                    {hx, hy, hz},    {-hx, hy, hz}};

  std::vector<cv::Point2f> pts2d;
  cv::projectPoints(pts3d, p.rvec(), p.tvec(), camera_param.cameraMatrix,
                    camera_param.distCoeff, pts2d);

  for (int i = 0; i < 4; i++)
  {
    cv::line(img, pts2d[i], pts2d[(i + 1) % 4], color, thickness);
    cv::line(img, pts2d[i + 4], pts2d[(i + 1) % 4 + 4], color, thickness);
    cv::line(img, pts2d[i], pts2d[i + 4], color, thickness);
  }
}

void RuneDetectorNode::DrawAxes(cv::Mat& img, const PoseNode& p, float length,
                                int thickness)
{
  std::vector<cv::Point3f> axes_pts = {
      {0, 0, 0}, {length, 0, 0}, {0, length, 0}, {0, 0, length}};
  std::vector<cv::Point2f> axes_proj;
  cv::projectPoints(axes_pts, p.rvec(), p.tvec(), camera_param.cameraMatrix,
                    camera_param.distCoeff, axes_proj);

  cv::line(img, axes_proj[0], axes_proj[1], cv::Scalar(0, 0, 255),
           thickness);  // X: Red
  cv::line(img, axes_proj[0], axes_proj[2], cv::Scalar(0, 255, 0),
           thickness);  // Y: Green
  cv::line(img, axes_proj[0], axes_proj[3], cv::Scalar(255, 0, 0),
           thickness);  // Z: Blue
}

void RuneDetectorNode::DrawPredictedPoint(cv::Mat& img,
                                          const std::shared_ptr<RuneGroup>& rune_group)
{
  auto trackers = rune_group->getTrackers();
  for (const auto& tracker_node : trackers)
  {
    auto tracker = RuneTracker::cast(tracker_node);
    if (!tracker || tracker->getHistoryNodes().empty())
    {
      continue;
    }
    auto combo = RuneCombo::cast(tracker->getHistoryNodes().front());
    if (!combo)
    {
      continue;
    }
    if (combo->getRuneType() != RuneType::PENDING_STRUCK)
    {
      continue;
    }

    // 对 PENDING_STRUCK 的目标画一个黄色菱形标记
    if (combo->getImageCache().isSetCenter())
    {
      cv::Point2f center = combo->getImageCache().getCenter();
      cv::drawMarker(img, center, cv::Scalar(0, 255, 255), cv::MARKER_DIAMOND, 25, 3);

      // 如果有预测函数, 可以绘制预测点
      auto predict_func = rune_group->getPredictFunc();
      if (predict_func)
      {
        int64_t now_tick = rune_group->getHistoryTicks().empty()
                               ? 0
                               : rune_group->getHistoryTicks().front();
        // 预测 200ms 后的位置
        // ... (需要将角度预测转换为2D投影, 此处省略复杂实现)
        cv::putText(img, "PREDICT", center + cv::Point2f(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 200, 255), 1);
      }
    }
  }
}

// ============================================================================
// 参数声明
// ============================================================================
void RuneDetectorNode::DeclareParameters()
{
  // 基础参数
  this->declare_parameter<int>("detect_color", 0);
  this->declare_parameter<int>("color_threshold", 100);
  this->declare_parameter<bool>("debug", false);

  // 坐标系参数 (已根据 URDF 调整默认值)
  this->declare_parameter<std::string>("target_frame", "gimbal_odom");
  this->declare_parameter<std::string>("source_frame", "pitch_link");
  this->declare_parameter<std::string>("camera_frame", "camera_optical_frame");

  // RuneDetector 参数
  this->declare_parameter<double>("detector.min_contour_area", 20.0);
  this->declare_parameter<double>("detector.max_contour_area", 20000.0);
  this->declare_parameter<double>("detector.big_active_fan_area", 1000.0);
  this->declare_parameter<double>("detector.min_center_accuracy", 0.80);
  this->declare_parameter<double>("detector.max_distance_ratio", 1.5);
  this->declare_parameter<double>("detector.max_match_deviation_ratio", 0.2);
  this->declare_parameter<bool>("detector.enable_center_force_construct", true);
  this->declare_parameter<double>("detector.center_force_construct_ratio", 0.2);

  // RuneGroup 参数
  this->declare_parameter<int>("group.max_vanish_number", 5);
  this->declare_parameter<double>("group.max_distance", 12000.0);
  this->declare_parameter<double>("group.min_distance", 3000.0);
  this->declare_parameter<double>("group.max_x_deviation", 1000.0);
  this->declare_parameter<double>("group.max_y_deviation", 1000.0);
  this->declare_parameter<double>("group.max_z_deviation", 1000.0);
  this->declare_parameter<double>("group.max_yaw_deviation", 10.0);
  this->declare_parameter<double>("group.max_pitch_deviation", 10.0);
  this->declare_parameter<double>("group.max_roll_deviation", 10.0);
  this->declare_parameter<bool>("group.enable_extreme_value_filter", true);
  this->declare_parameter<int>("group.raw_datas_size", 500);

  // 物理尺寸参数 (mm)
  this->declare_parameter<double>("rune.center_translation_z", -165.44);
  this->declare_parameter<double>("rune.target_translation_y", -700.0);
}

// ============================================================================
// 参数同步
// ============================================================================
void RuneDetectorNode::SyncParametersToCore()
{
  rune_detector_param.MIN_CONTOUR_AREA =
      this->get_parameter("detector.min_contour_area").as_double();
  rune_detector_param.MAX_CONTOUR_AREA =
      this->get_parameter("detector.max_contour_area").as_double();
  rune_detector_param.BIG_ACTIVE_FAN_AREA =
      this->get_parameter("detector.big_active_fan_area").as_double();
  rune_detector_param.MIN_CENTER_ACCURACY =
      this->get_parameter("detector.min_center_accuracy").as_double();
  rune_detector_param.MAX_DISTANCE_RATIO =
      this->get_parameter("detector.max_distance_ratio").as_double();
  rune_detector_param.MAX_MATCH_DEVIATION_RATIO =
      this->get_parameter("detector.max_match_deviation_ratio").as_double();
  rune_detector_param.ENABLE_CENTER_FORCE_CONSTRUCT_WINDOW =
      this->get_parameter("detector.enable_center_force_construct").as_bool();
  rune_detector_param.CENTER_FORCE_CONSTRUCT_WINDOW_RATIO =
      this->get_parameter("detector.center_force_construct_ratio").as_double();

  rune_group_param.MAX_VANISH_NUMBER =
      static_cast<int>(this->get_parameter("group.max_vanish_number").as_int());
  rune_group_param.MAX_DISTANCE = this->get_parameter("group.max_distance").as_double();
  rune_group_param.MIN_DISTANCE = this->get_parameter("group.min_distance").as_double();
  rune_group_param.MAX_X_DEVIATION =
      this->get_parameter("group.max_x_deviation").as_double();
  rune_group_param.MAX_Y_DEVIATION =
      this->get_parameter("group.max_y_deviation").as_double();
  rune_group_param.MAX_Z_DEVIATION =
      this->get_parameter("group.max_z_deviation").as_double();
  rune_group_param.MAX_YAW_DEVIATION =
      this->get_parameter("group.max_yaw_deviation").as_double();
  rune_group_param.MAX_PITCH_DEVIATION =
      this->get_parameter("group.max_pitch_deviation").as_double();
  rune_group_param.MAX_ROLL_DEVIATION =
      this->get_parameter("group.max_roll_deviation").as_double();
  rune_group_param.ENABLE_EXTREME_VALUE_FILTER =
      this->get_parameter("group.enable_extreme_value_filter").as_bool();
  rune_group_param.RAW_DATAS_SIZE =
      static_cast<size_t>(this->get_parameter("group.raw_datas_size").as_int());

  rune_center_param.TRANSLATION =
      cv::Matx31d(0, 0, this->get_parameter("rune.center_translation_z").as_double());
  rune_target_param.TRANSLATION =
      cv::Matx31d(0, this->get_parameter("rune.target_translation_y").as_double(), 0);
}

}  // namespace rm_rune_detector

// ============================================================================
// 组件注册
// ============================================================================
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(rm_rune_detector::RuneDetectorNode)