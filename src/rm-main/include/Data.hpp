#ifndef INCLUDE_DATA_HPP
#define INCLUDE_DATA_HPP
#include <cstdint>
#include <chrono>
#include "opencv2/core/quaternion.hpp"
#include "opencv2/core/mat.hpp"
#include "RTSerial.hpp"

struct __attribute__((packed)) Packet{

    uint8_t header;       // 0xA5
    uint8_t target_id;    // 0xEA
    uint8_t length;       // 0x1A (26)
    uint8_t cmd_id;       // 0x35
    uint8_t head_chk;     // 0xA6
    uint32_t timestamp;   // 0x7A100000 (Little Endian) or ID
    float q0;             // x
    float q1;             // y
    float q2;             // z
    float q3;             // w
    uint8_t checksum;     // 校验和
} ;
struct __attribute__((packed)) ShootPosi{

    uint8_t header = 0xA5;       // 0xA5
    uint8_t id_1 = 0xCB;    // 0xEA
    uint8_t id_2 = 0x86 ;       // 0x1A (26)
    uint8_t id_3 = 0x09 ;       // 0x35
    uint8_t id_4 = 0x6F ;     // 0xA6
    uint8_t len_1 = 0x00 ;   // 0x7A100000 (Little Endian) or ID
    uint8_t len_2 = 0x00 ;
    uint8_t len_3 = 0x0C;
    uint8_t crc_head = 0x55;
    
    float row = 0.0; 
    float pitch;            // x
    float yaw;             // y
            
    uint8_t checksum;     // 校验和

    ShootPosi(float pitch, float yaw): pitch(pitch), yaw(yaw){ this->checksum = io::CRC8::Calculate(this, sizeof(*this)-1); }
};

struct __attribute__((packed)) ShootFire{

    uint8_t header = 0xA5;       // 0xA5
    uint8_t id_1 = 0x6B;    // 0xEA
    uint8_t id_2 = 0xAD ;       // 0x1A (26)
    uint8_t id_3 = 0x91 ;       // 0x35
    uint8_t id_4 = 0x02 ;;     // 0xA6
    uint8_t len_1 =  0x00 ;   // 0x7A100000 (Little Endian) or ID
    uint8_t len_2 = 0x00 ;
    uint8_t len_3 = 0x01;
    uint8_t crc_head = 0x1A;

    bool fire;             // z

    uint8_t checksum;     // 校验和

    ShootFire(bool fire): fire(fire) { this->checksum = io::CRC8::Calculate(this, sizeof(*this)-1); }
};


struct FrameData
{
    cv::Mat image;
    cv::Quatd quat;
    std::chrono::steady_clock::time_point time;

    FrameData(const cv::Mat image, const cv::Quatd& quat,
              const std::chrono::steady_clock::time_point& time)
        : image(image), quat(quat), time(time) {}
    FrameData(){}
};

#endif