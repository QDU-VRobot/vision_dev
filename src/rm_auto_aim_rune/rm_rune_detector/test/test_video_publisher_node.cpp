/**
 * @file test_video_publisher_node.cpp
 * @brief 视频帧发布节点
 * @details 将视频文件逐帧发布为 sensor_msgs/Image 和 CameraInfo 话题，
 *          配合 rune_detector_node 使用，测试完整的 ROS2 检测管线。
 *          需要同时启动 test_mock_tf_node 提供 TF 变换。
 *
 * 用法: ros2 run rm_rune_detector test_video_publisher_node
 *       --ros-args -p video_path:=<path>
 */

#include <cv_bridge/cv_bridge.h>

#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

class VideoPublisherNode : public rclcpp::Node
{
 public:
  VideoPublisherNode() : Node("video_publisher_node")
  {
    this->declare_parameter<std::string>("video_path", "");
    this->declare_parameter<double>("fx", 1280.0);
    this->declare_parameter<double>("fy", 1280.0);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 512.0);
    this->declare_parameter<double>("publish_rate", 30.0);
    this->declare_parameter<bool>("loop", true);

    std::string video_path = this->get_parameter("video_path").as_string();
    if (video_path.empty())
    {
      RCLCPP_ERROR(get_logger(), "请通过 -p video_path:=<path> 指定视频路径");
      return;
    }

    cap_.open(video_path);
    if (!cap_.isOpened())
    {
      RCLCPP_ERROR(get_logger(), "无法打开视频: %s", video_path.c_str());
      return;
    }

    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    RCLCPP_INFO(get_logger(), "视频: %s (%dx%d)", video_path.c_str(), width_, height_);

    img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/image_raw",
                                                               rclcpp::SensorDataQoS());
    cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS());

    double rate = this->get_parameter("publish_rate").as_double();
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
        std::bind(&VideoPublisherNode::PublishFrame, this));
  }

 private:
  void PublishFrame()
  {
    cv::Mat frame;
    cap_.read(frame);
    if (frame.empty())
    {
      if (this->get_parameter("loop").as_bool())
      {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        cap_.read(frame);
        if (frame.empty())
        {
          return;
        }
        RCLCPP_INFO(get_logger(), "视频循环播放");
      }
      else
      {
        RCLCPP_INFO(get_logger(), "视频结束");
        timer_->cancel();
        return;
      }
    }

    auto now = this->now();

    // 发布图像
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = now;
    msg->header.frame_id = "camera_optical_frame";
    img_pub_->publish(*msg);

    // 发布相机信息
    sensor_msgs::msg::CameraInfo info;
    info.header.stamp = now;
    info.header.frame_id = "camera_optical_frame";
    info.width = width_;
    info.height = height_;
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

  cv::VideoCapture cap_;
  int width_ = 0, height_ = 0;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VideoPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
