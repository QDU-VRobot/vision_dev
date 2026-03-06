#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <Eigen/Geometry>
#include <sys/types.h>

// LibXR
#include "SharedTopic.hpp"
#include "SharedTopicClient.hpp"
#include "app_framework.hpp"
#include "linux_uart.hpp"
#include "message.hpp"
#include "thread.hpp"

/*消息包*/
// 云台欧拉角数据结构体
typedef struct {
  float pitch;
  float yaw;
  float roll;
} gimbal_euler;

static void XRobotMain(LibXR::HardwareContainer &hw) {
  using namespace LibXR;
  static ApplicationManager appmgr;

  static SharedTopic shared_topic(hw, appmgr, "uart_client", 81920, 256,
                                  {{"ahrs_quaternion"}});

  static SharedTopicClient shared_topic_client(hw, appmgr, "uart_client", 81920,
                                               256, {{"target_euler"}});
  static SharedTopicClient shared_topic_client_1(
      hw, appmgr, "uart_client", 81920, 256, {{"fire_notify", "tracker"}});
}

/* RMSerialDriver类定义*/
class RMSerialDriver {
public:
  RMSerialDriver();
  ~RMSerialDriver() = default;

  double timestamp_offset{};

  int ahrs_receive_cnt = 0;
  int ahrs_print_freq = 50;

//private:
  // uint8_t stop_notify_0 = 0;
  // uint8_t stop_notify_1 = 1;
  float yaw_deflection;
  uint8_t fire_notify = 0;
  /* 函数声明 */

  // 四元数转欧拉角函数
  void ConvertQuaternionToEuler(float qx, float qy, float qz, float qw,
                                float &roll, float &pitch, float &yaw);

  /* LibXR Topic (用于发送到下位机) */
  LibXR::Topic gimbal_cmd;
  LibXR::Topic launcher_cmd;
  // LibXR::Topic stop_notify_topic_;

  /* LibXR初始化相关成员变量 */
  std::unique_ptr<LibXR::RamFS> ramfs_;
  std::unique_ptr<LibXR::LinuxUART> uart_client_;
  std::unique_ptr<LibXR::Terminal<1024, 64, 16, 128>> terminal_;
  std::unique_ptr<LibXR::Thread> term_thread_;
  std::unique_ptr<LibXR::HardwareContainer> peripherals_;
private:
};
#endif // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
