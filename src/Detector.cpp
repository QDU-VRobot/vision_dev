#include "../include/HikCamera.hpp"
#include "../include/serial_driver.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

using namespace cv;
using namespace std;

int main() {
  io::HikCamera Hik(0.3, 10); // 0.5,3  0.0001,0.01
  Hik.continueCap(5);

  // while (true)
  // {
  //     io::HikCamera::ImageData frame;
  //     Hik.read(frame);
  //     cv::imread("photo", frame.image);
  //     cv::namedWindow("photo", cv::WINDOW_AUTOSIZE);
  //     cv::imshow("photo", frame.image);
  //     cv::waitKey(1);
  // }
  // VideoCapture cap(0);
  // if (!cap.isOpened())
  // {
  //     cout << "摄像头打开失败" << endl;
  //     return -1;
  // }

  RMSerialDriver driver;
  std::cerr << "Serial driver initialized" << std::endl;

  Mat hsv, mask, v_channel;

  bool useRGB = true; // true 时直接在 BGR 空间阈值，false 时使用原来的 HSV 分离

  while (true) {
    io::HikCamera::ImageData frame;
    Hik.read(frame);

    // auto start_time = chrono::high_resolution_clock::now();

    // ===== 颜色分割 =====
    if (useRGB) {
      // 直接对 BGR 图像二值化
      // 注意 OpenCV 是 BGR 顺序，下面的阈值只是示例，
      // 需要根据实际环境调节。
      Scalar lower_bgr(0, 100, 0); // B,G,R
      Scalar upper_bgr(80, 255, 80);
      inRange(frame.image, lower_bgr, upper_bgr, mask);

      // 或者更精细的做法，比较 G 通道与其他两个通道：
      /*
      Mat bgr[3];
      split(frame.image, bgr);
      Mat greenOnly = bgr[1] > bgr[0];
      greenOnly &= (bgr[1] > bgr[2]);
      greenOnly &= (bgr[1] > 120);    // 最低绿光强度
      mask = greenOnly;
      */
    } else {
      // ===== 转HSV =====
      cvtColor(frame.image, hsv, COLOR_BGR2HSV);

      // 分离V通道（亮度）
      vector<Mat> hsv_split;
      split(hsv, hsv_split);
      v_channel = hsv_split[2];

      // ===== 提取绿色 =====
      Scalar lower_green(35, 60, 80);
      Scalar upper_green(85, 255, 255);
      inRange(hsv, lower_green, upper_green, mask);
    }

    // ===== 亮度过滤（灯光更亮）=====
    // Mat bright_mask;
    // threshold(v_channel, bright_mask, 150, 255, THRESH_BINARY);

    // bitwise_and(mask, bright_mask, mask);

    // namedWindow("亮度过滤",WINDOW_NORMAL);
    // resizeWindow("亮度过滤",640,512);
    // moveWindow("亮度过滤",1700,0);
    // imshow("亮度过滤",mask);

    // ===== 去噪 =====
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // ===== 查找轮廓 =====
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto &contour : contours) {
      double area = contourArea(contour);
      if (area < 150)
        continue; // 小区域过滤

      // ===== 圆形度判断 =====
      double perimeter = arcLength(contour, true);
      if (perimeter == 0)
        continue;

      double circularity = 4 * CV_PI * area / (perimeter * perimeter);

      if (circularity < 0.6)
        continue;

      // ===== 椭圆拟合 =====
      if (contour.size() < 5)
        continue;

      RotatedRect ellipse_rect = fitEllipse(contour);

      float w = ellipse_rect.size.width;
      float h = ellipse_rect.size.height;
      float ratio = max(w, h) / min(w, h);

      // 长宽比过滤（接近圆/椭圆）
      if (ratio > 1.8)
        continue;

      // ===== 绘制检测结果 =====
      ellipse(frame.image, ellipse_rect, Scalar(0, 0, 255), 2);

      circle(frame.image, ellipse_rect.center, 3, Scalar(255, 0, 0), -1);

      // if (ellipse_rect.center.x > 600 && ellipse_rect.center.x < 680)
      //     {
      //         // 发送停止指令
      //         driver.stop_notify_topic_.Publish(driver.stop_notify_1);
      //     }
      // else
      //     {
      //         // 发送继续指令
      //         driver.stop_notify_topic_.Publish(driver.stop_notify_0);
      //     }

      // if (ellipse_rect.center.x < 640)
      //     {
      //         // 发送左转指令
      //         driver.yaw_deflection = (ellipse_rect.center.x-640)/2;
      //         driver.yaw_deflection_topic_.Publish(driver.yaw_deflection);
      //     }
      // else
      //     {
      //         // 发送右转指令
      //         driver.yaw_deflection = (ellipse_rect.center.x-640)/2;
      //         driver.yaw_deflection_topic_.Publish(driver.yaw_deflection);
      //     }
      driver.yaw_deflection =
          (ellipse_rect.center.x - 640) / 10; //修改除数以调整转向灵敏度
      std::cout << "Yaw deflection: " << driver.yaw_deflection << std::endl;
      driver.host_dart_gimbal_cmd.Publish(driver.yaw_deflection);

      if (ellipse_rect.center.x > 639 && ellipse_rect.center.x < 641) {
        driver.fire_notify_0 = 1;
      } else {
        driver.fire_notify_0 = 0;
      }

      std ::cout << "Fire notify: " << std::to_string(driver.fire_notify_0) << std::endl;
      driver.fire_notify.Publish(driver.fire_notify_0);
      // 1280 1024
    }

    namedWindow("Green Light Detection", WINDOW_NORMAL);
    resizeWindow("Green Light Detection", 640, 512);
    moveWindow("Green Light Detection", 100, 0);
    imshow("Green Light Detection", frame.image);

    namedWindow("Mask", WINDOW_NORMAL);
    resizeWindow("Mask", 640, 512);
    moveWindow("Mask", 1000, 0);
    imshow("Mask", mask);

    if (waitKey(30) == 27)
      break;
    
      auto end_time = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
      cout << "Processing time: " << duration.count() << " ms" << endl;
  }

  // cap.release();
  destroyAllWindows();
  return 0;
}
 // if (waitKey(30) == 27)
    //   break;