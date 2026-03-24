#ifndef TRACKER_HPP
#define TRACKER_HPP
#include "Armor.hpp"
#include "Target.hpp"
#include <array>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/core/quaternion.hpp>
#include <vector>

class Tracker
{
public:
    // 状态机枚举
    enum class State {
        Searching,      // 搜索状态
        Tracking,       // 追踪状态
        TempLost,       // 暂丢状态
        Lost            // 彻底丢失状态
    };

    // 构造函数，可配置阈值参数
    Tracker(size_t search_threshold = 20,
            size_t temp_lost_threshold = 10,
            size_t lost_threshold = 30,
            size_t switch_threshold = 8);

    // 主处理函数
    void operator()(std::vector<ArmorPosi>& armors_posi,
                    const cv::Quatd& gripper_to_world,
                    const Eigen::Matrix<double, 3, 1>& Gun, 
                    double dt);

    // 获取当前追踪的机器人
    Robot* getCurrentRobot() { return current_robot; }

    // 获取当前状态
    State getState() const { return current_state; }

    // 重置追踪器
    void reset();

private:
    // 状态转换函数
    void handleSearching(std::vector<ArmorPosi>& armors_posi,
                        const cv::Quatd& gripper_to_world,
                        const Eigen::Matrix<double, 3, 1>& Gun,
                        double dt);

    void handleTracking(std::vector<ArmorPosi>& armors_posi,
                       const cv::Quatd& gripper_to_world,
                       const Eigen::Matrix<double, 3, 1>& Gun,
                       double dt);

    void handleTempLost(std::vector<ArmorPosi>& armors_posi,
                       const cv::Quatd& gripper_to_world,
                       double dt);

    // 辅助函数
    int MaxCloseCenterTarget(const std::vector<ArmorPosi>& armors_posi,const Eigen::Matrix<double, 3, 1>& Gun);

    bool isTargetInCenter(const ArmorPosi& armor, double threshold = 0.3);

    ArmorPosi::Type findMostFrequentType(const std::vector<ArmorPosi>& armors);

    std::vector<ArmorPosi> filterByType(const std::vector<ArmorPosi>& armors,
                                        ArmorPosi::Type type);

    // 目标计数：[hero, infantry, sentry] 等
    static std::array<size_t, 7> Num;

    // 当前状态
    State current_state;

    // 当前追踪的机器人
    Robot* current_robot;
    Robot robot_instance;

    // 当前追踪的目标类型
    ArmorPosi::Type current_target_type;

    // 连续丢失计数
    size_t lost_count;

    // 新目标在中心的连续帧数
    size_t center_target_count;
    ArmorPosi::Type center_target_type;

    // 搜索阶段收集的数据
    std::array<std::vector<std::vector<ArmorPosi>>, 7> search_data_buffers;
    std::array<std::vector<double>, 7> search_time_buffers;

    // 阈值配置
    size_t search_threshold;      // 搜索状态进入追踪的阈值
    size_t temp_lost_threshold;   // 进入暂丢状态的阈值
    size_t lost_threshold;        // 彻底丢失的阈值
    size_t switch_threshold;      // 目标切换的阈值
};


#endif // TRACKER_HPP