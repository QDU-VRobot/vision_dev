/**
 * @file test_image_node.cpp
 * @brief 单张图片能量机关识别测试节点
 * @details 加载一张图片，执行检测，输出结果到调试图像窗口和终端。
 *          无需陀螺仪数据、无需 tf2。
 *
 * 用法: ros2 run rm_rune_detector test_image_node
 *       --ros-args -p image_path:=<path> -p detect_color:=0
 */

#include <cv_bridge/cv_bridge.h>
#include <vc/camera/camera_param.h>
#include <vc/detector/rune_detector.h>
#include <vc/detector/rune_detector_param.h>
#include <vc/feature/rune_center_param.h>
#include <vc/feature/rune_combo.h>
#include <vc/feature/rune_group.h>
#include <vc/feature/rune_group_param.h>
#include <vc/feature/rune_target_param.h>
#include <vc/feature/rune_tracker.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class TestImageNode : public rclcpp::Node
{
 public:
  TestImageNode() : Node("test_image_node")
  {
    this->declare_parameter<std::string>("image_path", "");
    this->declare_parameter<int>("detect_color", 0);
    this->declare_parameter<int>("color_threshold", 100);
    // 简单相机参数 (可通过参数覆盖)
    this->declare_parameter<double>("fx", 1280.0);
    this->declare_parameter<double>("fy", 1280.0);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 512.0);
    this->declare_parameter<bool>("show_window", true);
    this->declare_parameter<std::string>("output_path", "");

    Run();
  }

 private:
  void Run()
  {
    std::string img_path = this->get_parameter("image_path").as_string();
    if (img_path.empty())
    {
      RCLCPP_ERROR(get_logger(), "请通过 -p image_path:=<path> 指定图片路径");
      return;
    }

    cv::Mat frame = cv::imread(img_path);
    if (frame.empty())
    {
      RCLCPP_ERROR(get_logger(), "无法读取图片: %s", img_path.c_str());
      return;
    }
    RCLCPP_INFO(get_logger(), "加载图片: %s (%dx%d)", img_path.c_str(), frame.cols,
                frame.rows);

    // 设置相机参数
    float fx = static_cast<float>(this->get_parameter("fx").as_double());
    float fy = static_cast<float>(this->get_parameter("fy").as_double());
    float cx = static_cast<float>(this->get_parameter("cx").as_double());
    float cy = static_cast<float>(this->get_parameter("cy").as_double());
    camera_param.cameraMatrix = cv::Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    camera_param.distCoeff = cv::Matx<float, 5, 1>(0, 0, 0, 0, 0);
    camera_param.image_width = frame.cols;
    camera_param.image_height = frame.rows;
    camera_param.cam2joint_rmat = cv::Matx33f::eye();
    camera_param.cam2joint_tvec = cv::Matx<float, 3, 1>(0, 0, 0);

    int detect_color = static_cast<int>(this->get_parameter("detect_color").as_int());
    int threshold = static_cast<int>(this->get_parameter("color_threshold").as_int());

    // 创建检测器
    auto detector = RuneDetector::make_detector();
    std::vector<FeatureNode_ptr> rune_groups;

    DetectorInput input;
    DetectorOutput output;
    input.setImage(frame);
    input.setGyroData(GyroData());
    input.setTick(cv::getTickCount());
    input.setColor(detect_color == 0 ? PixChannel::RED : PixChannel::BLUE);
    input.setColorThresh(threshold);
    input.setFeatureNodes(rune_groups);

    auto t1 = std::chrono::steady_clock::now();
    detector->detect(input, output);
    auto t2 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    rune_groups = output.getFeatureNodes();

    RCLCPP_INFO(get_logger(), "=== 检测结果 ===");
    RCLCPP_INFO(get_logger(), "耗时: %.2f ms", ms);
    RCLCPP_INFO(get_logger(), "有效: %s", output.getValid() ? "YES" : "NO");
    RCLCPP_INFO(get_logger(), "RuneGroup 数量: %zu", rune_groups.size());

    cv::Mat debug_img = frame.clone();

    if (!rune_groups.empty())
    {
      auto rune_group = RuneGroup::cast(rune_groups.front());
      if (rune_group)
      {
        rune_group->drawFeature(debug_img);

        auto trackers = rune_group->getTrackers();
        int n = 0;
        for (auto& t : trackers)
        {
          auto tk = RuneTracker::cast(t);
          if (!tk || tk->getHistoryNodes().empty())
          {
            continue;
          }
          auto combo = RuneCombo::cast(tk->getHistoryNodes().front());
          if (!combo)
          {
            continue;
          }

          std::string type_str;
          switch (combo->getRuneType())
          {
            case RuneType::STRUCK:
              type_str = "STRUCK";
              break;
            case RuneType::UNSTRUCK:
              type_str = "UNSTRUCK";
              break;
            case RuneType::PENDING_STRUCK:
              type_str = "PENDING";
              break;
            default:
              type_str = "UNKNOWN";
              break;
          }
          RCLCPP_INFO(get_logger(), "  Tracker[%d]: %s, drop=%d", n++, type_str.c_str(),
                      tk->getDropFrameCount());
        }

        float angle = 0;
        if (rune_group->getCurrentRotateAngle(angle))
        {
          RCLCPP_INFO(get_logger(), "  转角: %.1f deg", angle);
        }

        if (rune_group->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
        {
          auto& p = rune_group->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);
          RCLCPP_INFO(get_logger(), "  中心位置 (mm): [%.0f, %.0f, %.0f]", p.tvec()(0),
                      p.tvec()(1), p.tvec()(2));
          RCLCPP_INFO(get_logger(), "  距离: %.0f mm", cv::norm(p.tvec()));
        }
      }
    }

    // 画信息覆盖
    cv::putText(debug_img,
                "Detect: " + std::string(output.getValid() ? "YES" : "NO") + " | " +
                    std::to_string(static_cast<int>(ms)) + "ms",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0),
                2);

    // 输出路径
    std::string output_path = this->get_parameter("output_path").as_string();
    if (!output_path.empty())
    {
      cv::imwrite(output_path, debug_img);
      RCLCPP_INFO(get_logger(), "结果保存到: %s", output_path.c_str());
    }

    // 显示窗口
    if (this->get_parameter("show_window").as_bool())
    {
      cv::imshow("Rune Detection - Image Test", debug_img);
      RCLCPP_INFO(get_logger(), "按任意键退出...");
      cv::waitKey(0);
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TestImageNode>();
  rclcpp::shutdown();
  return 0;
}
