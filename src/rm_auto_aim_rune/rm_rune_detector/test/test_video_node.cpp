/**
 * @file test_video_node.cpp
 * @brief 视频能量机关识别与追踪测试节点
 * @details 加载视频文件，逐帧执行检测与追踪，显示实时调试图像。
 *          无需陀螺仪数据、无需 tf2。
 *
 * 用法: ros2 run rm_rune_detector test_video_node
 *       --ros-args -p video_path:=<path> -p detect_color:=0
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

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>

class TestVideoNode : public rclcpp::Node
{
 public:
  TestVideoNode() : Node("test_video_node")
  {
    this->declare_parameter<std::string>("video_path", "");
    this->declare_parameter<int>("detect_color", 0);
    this->declare_parameter<int>("color_threshold", 100);
    this->declare_parameter<double>("fx", 1280.0);
    this->declare_parameter<double>("fy", 1280.0);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 512.0);
    this->declare_parameter<bool>("show_window", true);
    this->declare_parameter<bool>("pause_on_detect", false);
    this->declare_parameter<double>("playback_speed", 1.0);
    this->declare_parameter<std::string>("output_video_path", "");

    Run();
  }

 private:
  inline void DrawCube(cv::Mat& img, const PoseNode& p, float x_len, float y_len,
                       float z_len, const cv::Scalar& color, int thickness = 1)
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

  inline void DrawAxes(cv::Mat& img, const PoseNode& p, float len, int t = 2)
  {
    std::vector<cv::Point3f> axes = {{0, 0, 0}, {len, 0, 0}, {0, len, 0}, {0, 0, len}};
    std::vector<cv::Point2f> proj;
    cv::projectPoints(axes, p.rvec(), p.tvec(), camera_param.cameraMatrix,
                      camera_param.distCoeff, proj);
    cv::line(img, proj[0], proj[1], cv::Scalar(0, 0, 255), t);
    cv::line(img, proj[0], proj[2], cv::Scalar(0, 255, 0), t);
    cv::line(img, proj[0], proj[3], cv::Scalar(255, 0, 0), t);
  }

  void Run()
  {
    std::string video_path = this->get_parameter("video_path").as_string();
    if (video_path.empty())
    {
      RCLCPP_ERROR(get_logger(), "请通过 -p video_path:=<path> 指定视频路径");
      return;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
      RCLCPP_ERROR(get_logger(), "无法打开视频: %s", video_path.c_str());
      return;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    RCLCPP_INFO(get_logger(), "视频: %s", video_path.c_str());
    RCLCPP_INFO(get_logger(), "  尺寸: %dx%d, 帧数: %d, FPS: %.1f", width, height,
                total_frames, fps);

    // 设置相机参数
    float fx = static_cast<float>(this->get_parameter("fx").as_double());
    float fy = static_cast<float>(this->get_parameter("fy").as_double());
    float cx = static_cast<float>(this->get_parameter("cx").as_double());
    float cy = static_cast<float>(this->get_parameter("cy").as_double());
    camera_param.cameraMatrix = cv::Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    camera_param.distCoeff = cv::Matx<float, 5, 1>(0, 0, 0, 0, 0);
    camera_param.image_width = width;
    camera_param.image_height = height;
    camera_param.cam2joint_rmat = cv::Matx33f::eye();
    camera_param.cam2joint_tvec = cv::Matx<float, 3, 1>(0, 0, 0);

    int detect_color = static_cast<int>(this->get_parameter("detect_color").as_int());
    int threshold = static_cast<int>(this->get_parameter("color_threshold").as_int());
    bool show_window = this->get_parameter("show_window").as_bool();
    bool pause_on_detect = this->get_parameter("pause_on_detect").as_bool();
    double playback_speed = this->get_parameter("playback_speed").as_double();

    // 输出视频
    std::string output_video_path = this->get_parameter("output_video_path").as_string();
    cv::VideoWriter writer;
    if (!output_video_path.empty())
    {
      writer.open(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                  fps > 0 ? fps : 30, cv::Size(width, height));
      if (writer.isOpened())
      {
        RCLCPP_INFO(get_logger(), "输出视频到: %s", output_video_path.c_str());
      }
    }

    auto detector = RuneDetector::make_detector();
    std::vector<FeatureNode_ptr> rune_groups;

    int frame_idx = 0;
    int detect_count = 0;
    double total_ms = 0;

    int wait_ms = (fps > 0 && playback_speed > 0)
                      ? static_cast<int>(1000.0 / fps / playback_speed)
                      : 30;

    cv::Mat frame;
    while (rclcpp::ok())
    {
      cap.read(frame);
      if (frame.empty())
      {
        RCLCPP_INFO(get_logger(), "视频结束。共处理 %d 帧, 检出 %d 帧, 平均耗时 %.1f ms",
                    frame_idx, detect_count, frame_idx > 0 ? total_ms / frame_idx : 0);
        break;
      }
      frame_idx++;

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
      total_ms += ms;

      rune_groups = output.getFeatureNodes();
      bool detected = output.getValid() && !rune_groups.empty();
      if (detected)
      {
        detect_count++;
      }

      // 绘制调试图像
      cv::Mat debug_img = frame.clone();
      if (detected)
      {
        auto rune_group = RuneGroup::cast(rune_groups.front());
        if (rune_group)
        {
          // 调用原始绘制
          rune_group->drawFeature(debug_img);

          // 额外: 为每个 tracker 绘制 3D 立方体
          for (auto& t_node : rune_group->getTrackers())
          {
            auto tk = RuneTracker::cast(t_node);
            if (!tk || tk->getHistoryNodes().empty())
            {
              continue;
            }
            auto combo = RuneCombo::cast(tk->getHistoryNodes().front());
            if (!combo)
            {
              continue;
            }

            cv::Scalar color;
            std::string label;
            switch (combo->getRuneType())
            {
              case RuneType::STRUCK:
                color = cv::Scalar(0, 255, 0);
                label = "STRUCK";
                break;
              case RuneType::PENDING_STRUCK:
                color = cv::Scalar(0, 255, 255);
                label = "PENDING";
                break;
              case RuneType::UNSTRUCK:
                color = cv::Scalar(255, 255, 255);
                label = "UNSTRUCK";
                break;
              default:
                color = cv::Scalar(128, 128, 128);
                label = "UNK";
                break;
            }

            if (tk->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
            {
              auto& p = tk->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);
              DrawCube(debug_img, p, 500, 500, 300, color, 2);
              DrawAxes(debug_img, p, 200, 1);
            }

            // 标记 PENDING 目标
            if (combo->getRuneType() == RuneType::PENDING_STRUCK &&
                combo->getImageCache().isSetCenter())
            {
              cv::drawMarker(debug_img, combo->getImageCache().getCenter(),
                             cv::Scalar(0, 255, 255), cv::MARKER_DIAMOND, 25, 3);
            }
          }

          // 中心坐标轴
          if (rune_group->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
          {
            auto& cp = rune_group->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);
            DrawAxes(debug_img, cp, 300, 2);
          }
        }
      }

      // 信息覆盖
      char buf[256];
      [[maybe_unused]] int res{};
      res = snprintf(buf, sizeof(buf), "Frame %d/%d | %.1fms | %s", frame_idx,
                     total_frames, ms, detected ? "DETECTED" : "---");
      cv::putText(debug_img, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

      if (detected && !rune_groups.empty())
      {
        auto rg = RuneGroup::cast(rune_groups.front());
        if (rg)
        {
          float angle = 0;
          rg->getCurrentRotateAngle(angle);
          res = snprintf(buf, sizeof(buf), "Angle: %.1f deg", angle);
          cv::putText(debug_img, buf, cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(0, 200, 255), 1);

          if (rg->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA))
          {
            double dist =
                cv::norm(rg->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA).tvec());
            res = snprintf(buf, sizeof(buf), "Dist: %.0f mm", dist);
            cv::putText(debug_img, buf, cv::Point(10, 78), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 200, 255), 1);
          }
        }
      }

      if (writer.isOpened())
      {
        writer.write(debug_img);
      }

      if (show_window)
      {
        cv::imshow("Rune Detection - Video Test", debug_img);
        int key = cv::waitKey(wait_ms);
        if (key == 'q' || key == 27)
        {
          break;  // ESC or q
        }
        if (key == ' ')
        {
          RCLCPP_INFO(get_logger(), "已暂停，按任意键继续...");
          cv::waitKey(0);
        }
      }

      if (pause_on_detect && detected)
      {
        if (show_window)
        {
          RCLCPP_INFO(get_logger(), "检测到目标，暂停。按任意键继续...");
          cv::waitKey(0);
        }
      }
    }

    if (writer.isOpened())
    {
      writer.release();
      RCLCPP_INFO(get_logger(), "输出视频已保存");
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TestVideoNode>();
  rclcpp::shutdown();
  return 0;
}
