#include "serial_driver.hpp"
#include <Eigen/Geometry>
#include <iostream>

using namespace std::chrono_literals;

RMSerialDriver::RMSerialDriver()
{
  LibXR::PlatformInit();
  peripherals_ = std::make_unique<LibXR::HardwareContainer>();
  ramfs_ = std::make_unique<LibXR::RamFS>();
  // 去除 ROS 参数声明，直接赋值
  std::string vid = "16d0";
  std::string pid = "1492";

  timestamp_offset = 0;
  std::cout << "Serial timestamp_offset: " << timestamp_offset << '\n';
  uart_client_ = std::make_unique<LibXR::LinuxUART>(
      vid, pid, 115200, LibXR::LinuxUART::Parity::NO_PARITY, 8, 1);
  terminal_ = std::make_unique<LibXR::Terminal<1024, 64, 16, 128>>(*ramfs_);
  term_thread_ = std::make_unique<LibXR::Thread>();
  term_thread_->Create(terminal_.get(), LibXR::Terminal<1024, 64, 16, 128>::ThreadFun,
                       "terminal", 81900, LibXR::Thread::Priority::MEDIUM);
  static LibXR::HardwareContainer peripherals{
      LibXR::Entry<LibXR::RamFS>({*ramfs_, {"ramfs"}}),
      LibXR::Entry<LibXR::UART>({*uart_client_, {"uart_client"}}),
  };

  // 从下位机接收的话题
  // ahrs_quaternion_topic_ =
  //     LibXR::Topic::FindOrCreate<LibXR::Quaternion<float>>("ahrs_quaternion");


  // // 发送到下位机的话题
  // LibXR::Topic::Domain tracker_domain = LibXR::Topic::Domain("tracker");
  gimbal_cmd =
      LibXR::Topic::FindOrCreate<uint8_t>("gimbal_cmd");
  launcher_cmd =
      LibXR::Topic::FindOrCreate<uint8_t>("launcher_cmd");
//   stop_notify_topic_ =
//       LibXR::Topic::FindOrCreate<uint8_t>("stop_notify", &tracker_domain);

  XRobotMain(peripherals);

  // 云台姿态回调
  void (*ahrs_quaternion_cb_fun)(bool, RMSerialDriver *self, LibXR::RawData &data) =
      [](bool, RMSerialDriver *self, LibXR::RawData &data)
  {
        auto quat = reinterpret_cast<LibXR::Quaternion<float> *>(data.addr_);

        gimbal_euler gimbal;
    self->ConvertQuaternionToEuler(quat->x(), quat->y(), quat->z(), quat->w(),
                                   gimbal.roll, gimbal.pitch, gimbal.yaw);
    if (++self->ahrs_receive_cnt % self->ahrs_print_freq == 0)
    {
          // you can print current gimbal data here
      std::cout<<"current gimbal: "<<gimbal.roll<<" "<<gimbal.pitch<<" "<<gimbal.yaw<<std::endl;
          self->ahrs_receive_cnt = 0;
        }
      };
  auto ahrs_quaternion_cb = LibXR::Topic::Callback::Create(ahrs_quaternion_cb_fun, this);
  gimbal_cmd.RegisterCallback(ahrs_quaternion_cb);

  // while (1)
  // {
  //   LibXR::Thread::Sleep(10); // 发送延迟，10ms
  // }
}

/*四元数转欧拉角*/
void RMSerialDriver::ConvertQuaternionToEuler(float qx, float qy, float qz,
                                              float qw, float &roll,
                                              float &pitch, float &yaw) {
  // Eigen::Quaternionf q(qw, qx, qy, qz); // Eigen 四元数顺序为 (w, x, y, z)
  // Eigen::Vector3f euler =
  //     q.toRotationMatrix().eulerAngles(0, 1, 2); // roll, pitch, yaw
  // roll = euler[0];
  // pitch = euler[1];
  // yaw = euler[2];

  //由于eulerAngels函数在Eigen库中被弃用，改用手动计算
  // roll (x-axis rotation)
  float sinr_cosp = 2.0f * (qw * qx + qy * qz);
  float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
  roll = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  float sinp = 2.0f * (qw * qy - qz * qx);
  if (std::fabs(sinp) >= 1.0f) {
    // 使用 90 度如果超出范围
    pitch = std::copysign(M_PI / 2.0f, sinp);
  } else {
    pitch = std::asin(sinp);
  }

  // yaw (z-axis rotation)
  float siny_cosp = 2.0f * (qw * qz + qx * qy);
  float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
  yaw = std::atan2(siny_cosp, cosy_cosp);
}