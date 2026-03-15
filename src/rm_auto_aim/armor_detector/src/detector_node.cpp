#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STL
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_node.hpp"

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions& options)
    : Node("armor_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode (Pure Pixel Mode)!");

  // Detector
  detector_ = InitDetector();

  // Armors Publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
      "/detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.125;
  armor_marker_.scale.z = 0.05;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.g = 0.5;
  armor_marker_.color.b = 1.0;
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/detector/marker", 10);

  // Debug Publishers
  debug_ = this->declare_parameter("debug", false);
  if (debug_)
  {
    CreateDebugPublishers();
  }

  // Debug param change moniter
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ = debug_param_sub_->add_parameter_callback(
      "debug",
      [this](const rclcpp::Parameter& p)
      {
        debug_ = p.as_bool();
        debug_ ? CreateDebugPublishers() : DestroyDebugPublishers();
      });

  // 【修复】：去除了参数中的 const 和 &，解决订阅报错；删除了 PnP 的初始化
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_info", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
      {
        cam_center_ = cv::Point2f(static_cast<float>(camera_info->k[2]),
                                  static_cast<float>(camera_info->k[5]));
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        cam_info_sub_.reset();
      });

  // 【修复】：去除了 bind，换成现代 Lambda 写法
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
        this->ImageCallback(img_msg);
      });
}

// 【修复】：去除了 const 和 &
void ArmorDetectorNode::ImageCallback(sensor_msgs::msg::Image::ConstSharedPtr IMG_MSG)
{
  auto armors = DetectArmors(IMG_MSG);

  if (cam_info_ != nullptr)
  {
    armors_msg_.header = armor_marker_.header = text_marker_.header = IMG_MSG->header;
    armors_msg_.armors.clear();
    marker_array_.markers.clear();
    armor_marker_.id = 0;
    text_marker_.id = 0;

    double fx = cam_info_->k[0];
    double fy = cam_info_->k[4];
    double cx = cam_info_->k[2];
    double cy = cam_info_->k[5];

    auto_aim_interfaces::msg::Armor armor_msg;
    for (const auto& armor : armors)
    {
      armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];

      // 【纯像素闭环魔法】：删掉所有的 PnP 代码，换成这一段直接映射的除法
      armor_msg.pose.position.x = (armor.center.x - cx) / fx;
      armor_msg.pose.position.y = (armor.center.y - cy) / fy;
      armor_msg.pose.position.z = 1.0;  // 固定深度，绝不跳变！

      armor_msg.pose.orientation.x = 0.0;
      armor_msg.pose.orientation.y = 0.0;
      armor_msg.pose.orientation.z = 0.0;
      armor_msg.pose.orientation.w = 1.0;

      armor_msg.distance_to_image_center = cv::norm(armor.center - cv::Point2f(cx, cy));

      // Fill the markers
      armor_marker_.id++;
      armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
      armor_marker_.pose = armor_msg.pose;
      
      text_marker_.id++;
      text_marker_.pose.position = armor_msg.pose.position;
      text_marker_.pose.position.y -= 0.1;
      
      armors_msg_.armors.emplace_back(armor_msg);
      marker_array_.markers.emplace_back(armor_marker_);
      marker_array_.markers.emplace_back(text_marker_);
    }

    // Publishing detected armors
    armors_pub_->publish(armors_msg_);

    // Publishing marker
    PublishMarkers();
  }
}

std::unique_ptr<Detector> ArmorDetectorNode::InitDetector()
{
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].step = 1;
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  
  // 【修复】：加上了 this-> 和 <int>，消除红字报错
  int binary_thres = this->declare_parameter<int>("binary_thres", 160, param_desc);

  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  int detect_color = this->declare_parameter<int>("detect_color", RED, param_desc);

  // 【修复】：加上了 this-> 和 <double>，彻底解决模板推导报错
  Detector::LightParams l_params = {
      .min_ratio = this->declare_parameter<double>("light.min_ratio", 0.02),
      .max_ratio = this->declare_parameter<double>("light.max_ratio", 2.0),
      .max_angle = this->declare_parameter<double>("light.max_angle", 40.0)};

  Detector::ArmorParams a_params = {
      .min_light_ratio = this->declare_parameter<double>("armor.min_light_ratio", 0.5),
      .min_small_center_distance =
          this->declare_parameter<double>("armor.min_small_center_distance", 0.4),
      .max_small_center_distance =
          this->declare_parameter<double>("armor.max_small_center_distance", 8.0),
      .max_angle = this->declare_parameter<double>("armor.max_angle", 45.0)};

  auto detector =
      std::make_unique<Detector>(binary_thres, detect_color, l_params, a_params);

  return detector;
}

// 【修复】：去除了 const 和 &
std::vector<Armor> ArmorDetectorNode::DetectArmors(sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

  detector_->binary_thres = static_cast<int>(get_parameter("binary_thres").as_int());
  detector_->detect_color = static_cast<int>(get_parameter("detect_color").as_int());

  auto armors = detector_->Detect(img);

  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  if (debug_)
  {
    binary_img_pub_.publish(
        cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img).toImageMsg());

    std::sort(detector_->debug_lights.data.begin(), detector_->debug_lights.data.end(),
              [](const auto& l1, const auto& l2) { return l1.center_x < l2.center_x; });
    std::sort(detector_->debug_armors.data.begin(), detector_->debug_armors.data.end(),
              [](const auto& a1, const auto& a2) { return a1.center_x < a2.center_x; });

    lights_data_pub_->publish(detector_->debug_lights);
    armors_data_pub_->publish(detector_->debug_armors);

    detector_->DrawResults(img);
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(
        cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
  }

  return armors;
}

void ArmorDetectorNode::CreateDebugPublishers()
{
  lights_data_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugLights>(
      "/detector/debug_lights", 10);
  armors_data_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugArmors>(
      "/detector/debug_armors", 10);

  binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
  result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::DestroyDebugPublishers()
{
  lights_data_pub_.reset(); 
  armors_data_pub_.reset();
  binary_img_pub_.shutdown(); 
  result_img_pub_.shutdown();
}

void ArmorDetectorNode::PublishMarkers()
{
  using Marker = visualization_msgs::msg::Marker;
  armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
  marker_array_.markers.emplace_back(armor_marker_);
  marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)