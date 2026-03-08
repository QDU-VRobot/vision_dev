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

static void XRobotMain(LibXR::HardwareContainer &hw) {
  using namespace LibXR;
  static ApplicationManager appmgr;

  static SharedTopicClient shared_topic_client(
      hw, appmgr, "uart_client", 81920, 256,
      {{"host_dart_gimbal_cmd"}, {"fire_notify"}});
}

/* RMSerialDriver类定义*/
class RMSerialDriver {
public:
  RMSerialDriver();
  ~RMSerialDriver() = default;

  double timestamp_offset{};

  float yaw_deflection;
  uint8_t fire_notify_0 = 0;

  /* LibXR Topic (用于发送到下位机) */
  LibXR::Topic host_dart_gimbal_cmd;
  LibXR::Topic fire_notify;

  /* LibXR初始化相关成员变量 */
  std::unique_ptr<LibXR::RamFS> ramfs_;
  std::unique_ptr<LibXR::LinuxUART> uart_client_;
  std::unique_ptr<LibXR::Terminal<1024, 64, 16, 128>> terminal_;
  std::unique_ptr<LibXR::Thread> term_thread_;
  std::unique_ptr<LibXR::HardwareContainer> peripherals_;

private:
};
#endif // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
