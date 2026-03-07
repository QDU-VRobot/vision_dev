#include "serial_driver.hpp"
#include <Eigen/Geometry>
#include <iostream>

using namespace std::chrono_literals;

RMSerialDriver::RMSerialDriver() {
  LibXR::PlatformInit();
  peripherals_ = std::make_unique<LibXR::HardwareContainer>();
  ramfs_ = std::make_unique<LibXR::RamFS>();
  std::string vid = "16d0";
  std::string pid = "1492";

  uart_client_ = std::make_unique<LibXR::LinuxUART>(
      vid, pid, 115200, LibXR::LinuxUART::Parity::NO_PARITY, 8, 1);
  terminal_ = std::make_unique<LibXR::Terminal<1024, 64, 16, 128>>(*ramfs_);
  term_thread_ = std::make_unique<LibXR::Thread>();
  term_thread_->Create(terminal_.get(),
                       LibXR::Terminal<1024, 64, 16, 128>::ThreadFun,
                       "terminal", 81900, LibXR::Thread::Priority::MEDIUM);
  
  static LibXR::HardwareContainer peripherals{
      LibXR::Entry<LibXR::RamFS>({*ramfs_, {"ramfs"}}),
      LibXR::Entry<LibXR::UART>({*uart_client_, {"uart_client"}}),
  };

  // // 发送到下位机的话题
  gimbal_cmd = LibXR::Topic::FindOrCreate<float>("gimbal_cmd");
  launcher_cmd = LibXR::Topic::FindOrCreate<uint8_t>("launcher_cmd");

  XRobotMain(peripherals);
}