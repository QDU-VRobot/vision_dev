#include "../include/Tracker.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

// 初始化静态成员
std::array<size_t, 7> Tracker::Num = {0, 0, 0, 0, 0, 0, 0};

Tracker::Tracker(size_t search_threshold,
                 size_t temp_lost_threshold,
                 size_t lost_threshold,
                 size_t switch_threshold)
    : search_threshold(search_threshold),
      temp_lost_threshold(temp_lost_threshold),
      lost_threshold(lost_threshold),
      switch_threshold(switch_threshold),
      current_state(State::Searching),
      current_robot(nullptr),
      current_target_type(ArmorPosi::Type::base),
      lost_count(0),
      center_target_count(0),
      center_target_type(ArmorPosi::Type::base)
{
}

void Tracker::operator()(std::vector<ArmorPosi>& armors_posi,
                         const cv::Quatd& gripper_to_world,
                         const Eigen::Matrix<double, 3, 1>& Gun,
                         double dt)
{
    switch (current_state)
    {
    case State::Searching:
        handleSearching(armors_posi, gripper_to_world, Gun, dt);
        break;

    case State::Tracking:
        handleTracking(armors_posi, gripper_to_world, Gun, dt);
        break;

    case State::TempLost:
        handleTempLost(armors_posi, gripper_to_world, dt);
        break;

    case State::Lost:
        // 彻底丢失状态，重新进入搜索
        this->reset();
        handleSearching(armors_posi, gripper_to_world, Gun, dt);
        break;
    }
}

void Tracker::handleSearching(std::vector<ArmorPosi>& armors_posi,
                              const cv::Quatd& gripper_to_world,
                              const Eigen::Matrix<double, 3, 1>& Gun,
                              double dt)
{
    if (armors_posi.empty())
    {
        // 没有目标，重置计数
        std::fill(Num.begin(), Num.end(), 0);
        for(int i=0;i<7;i++)
        {
            this->search_data_buffers[i].clear();
            this->search_time_buffers[i].clear();
        }
        return;
    }

    // 对每个装甲板类型进行计数
    std::array<int,7> hax_ = {0,0,0,0,0,0,0};
    std::array<std::vector<ArmorPosi>, 7> oneframe_data_buffers;
    for (const auto& armor : armors_posi)
    {
        int type_idx = static_cast<int>(armor.type);
        if (type_idx >= 0 && type_idx < static_cast<int>(Num.size()))
        {
            Num[type_idx]++;
            hax_[type_idx]++;
            oneframe_data_buffers[type_idx].push_back(armor);
        }
    }

    // 保存当前帧数据
    for(int i = 0;i<7;i++)
    {
        if(hax_[i] == 0)
        {
            Num[i] = 0;
            this->search_data_buffers[i].clear();
            this->search_time_buffers[i].clear();
        }
        else
        {
            this->search_data_buffers[i].push_back(std::move(oneframe_data_buffers[i]) );
            this->search_time_buffers[i].push_back(dt);
        }
    }

    // 检查是否有目标达到阈值
    for (size_t i = 0; i < Num.size(); ++i)
    {
        if (Num[i] >= search_threshold)
        {
            // 找到稳定目标，进入追踪状态
            current_target_type = static_cast<ArmorPosi::Type>(i);
            current_state = State::Tracking;
            current_robot = &this->robot_instance;

            std::cout << "[Tracker] 进入追踪状态，目标类型: " << i << "\n";

            current_robot->Init(this->search_data_buffers[i][0]);
            // 使用缓冲区中的所有数据初始化 Robot
            for (size_t j = 1; j < this->search_data_buffers[i].size(); ++j)
            {
                // i 是外部循环已经确定的目标类型索引，j 是帧序列号
                current_robot->Update(this->search_data_buffers[i][j], gripper_to_world, this->search_time_buffers[i][j]);
            }

            // 进入追踪模式，只有可能因为丢失而进入丢失状态
            lost_count = 0;

            return;
        }
    }
}

void Tracker::handleTracking(std::vector<ArmorPosi>& armors_posi,
                             const cv::Quatd& gripper_to_world,
                             const Eigen::Matrix<double, 3, 1>& Gun,
                             double dt)
{
    // 筛选出当前追踪类型的装甲板
    auto target_armors = filterByType(armors_posi, current_target_type);

    if (target_armors.empty())
    {
        // 没有检测到目标
        lost_count++;

        if (lost_count >= temp_lost_threshold)
        {
            // 进入暂丢状态
            current_state = State::TempLost;
            std::cout << "[Tracker] 进入暂丢状态" << std::endl;
            return ;
        }
        // 使用线性更新
        if (current_robot)
        {
            current_robot->Update( dt );
        }
    }
    else
    {
        // 检测到目标，重置丢失计数
        lost_count = 0;

        // 更新机器人状态
        if (current_robot)
        {
            current_robot->Update(target_armors, gripper_to_world, dt);
        }

        // 检查是否有新目标持续出现在中心
        if (!armors_posi.empty())
        {
            // 找到最靠近中心的装甲板
            const ArmorPosi* center_armor = nullptr;

            int id = this->MaxCloseCenterTarget(armors_posi, Gun);

            if (id != -1)
            {
                center_armor = &armors_posi[id];
            }
            
            
            if (center_armor != nullptr && center_armor->type != this->current_target_type)
            {
                // 有新目标在中心
                if (center_armor->type == center_target_type)
                {
                    center_target_count++;

                    if (center_target_count >= switch_threshold)
                    {
                        // 切换目标
                        std::cout << "[Tracker] 切换目标，新类型: "
                                  << static_cast<int>(center_armor->type) << "\n";

                        current_target_type = center_armor->type;
                        current_robot->Clear();

                        auto new_target_armors = filterByType(armors_posi, current_target_type);
                        if (!new_target_armors.empty())
                        {
                            current_robot->Init(new_target_armors);
                        }

                        center_target_count = 0;
                        lost_count = 0;
                    }
                }
                else
                {
                    // 新的中心目标类型
                    center_target_type = center_armor->type;
                    center_target_count = 1;
                }
            }
            else
            {
                // 没有新目标在中心，重置计数
                center_target_count = 0;
            }
        }
    }
}

void Tracker::handleTempLost(std::vector<ArmorPosi>& armors_posi,
                             const cv::Quatd& gripper_to_world,
                             double dt)
{
    // 筛选出当前追踪类型的装甲板
    auto target_armors = filterByType(armors_posi, current_target_type);

    if (target_armors.empty())
    {
        // 继续丢失
        lost_count++;

        if (lost_count >= lost_threshold)
        {
            // 彻底丢失
            current_state = State::Lost;
            std::cout << "[Tracker] 目标彻底丢失" << std::endl;

            return;
        }

        // 使用线性更新
        if (current_robot)
        {
            current_robot->Update(dt);
        }
    }
    else
    {
        // 重新找到目标，恢复追踪
        current_state = State::Tracking;
        lost_count = 0;
        std::cout << "[Tracker] 恢复追踪状态" << std::endl;

        // 更新机器人状态
        if (current_robot)
        {
            current_robot->Update(target_armors, gripper_to_world, dt);
        }
    }
}

int Tracker::MaxCloseCenterTarget(const std::vector<ArmorPosi>& armors_posi, const Eigen::Matrix<double, 3, 1>& Gun)
{
    // 通过判断向量点积，找到距离Gun向量最接近的装甲板ID
    if (armors_posi.empty())
    {
        return -1;
    }

    int best_idx = -1;
    double max_dot_product = -1.0;

    // 归一化枪管方向向量
    Eigen::Matrix<double, 3, 1> gun_normalized = Gun.normalized();

    for (size_t i = 0; i < armors_posi.size(); ++i)
    {
        // 构建装甲板位置向量
        Eigen::Matrix<double, 3, 1> armor_vec;
        armor_vec << armors_posi[i].posi.x,
                     armors_posi[i].posi.y,
                     armors_posi[i].posi.z;

        // 归一化装甲板方向向量
        Eigen::Matrix<double, 3, 1> armor_normalized = armor_vec.normalized();

        // 计算点积（余弦相似度）
        double dot_product = gun_normalized.dot(armor_normalized);

        // 找到点积最大的（方向最接近的）
        if (dot_product > max_dot_product)
        {
            max_dot_product = dot_product;
            best_idx = static_cast<int>(i);
        }
    }

    return best_idx;
}

bool Tracker::isTargetInCenter(const ArmorPosi& armor, double threshold)
{
    // 使用球坐标系的 phi 和 theta 判断是否在中心
    // phi 是方位角，theta 是仰角
    double phi = std::abs(armor.SCS.z);   // 方位角偏差
    double theta = std::abs(armor.SCS.y); // 仰角偏差

    return (phi < threshold && theta < threshold);
}

ArmorPosi::Type Tracker::findMostFrequentType(const std::vector<ArmorPosi>& armors)
{
    std::array<size_t, 7> type_count = {0};

    for (const auto& armor : armors)
    {
        int type_idx = static_cast<int>(armor.type);
        if (type_idx >= 0 && type_idx < static_cast<int>(type_count.size()))
        {
            type_count[type_idx]++;
        }
    }

    auto max_it = std::max_element(type_count.begin(), type_count.end());
    return static_cast<ArmorPosi::Type>(std::distance(type_count.begin(), max_it));
}

std::vector<ArmorPosi> Tracker::filterByType(const std::vector<ArmorPosi>& armors,
                                             ArmorPosi::Type type)
{
    std::vector<ArmorPosi> filtered;
    for (const auto& armor : armors)
    {
        if (armor.type == type)
        {
            filtered.push_back(armor);
        }
    }
    return filtered;
}

void Tracker::reset()
{
    if (current_robot)
    {
        current_robot->Clear();
    }
    current_state = State::Searching;
    current_robot = nullptr;
    lost_count = 0;
    center_target_count = 0;

    // 清空所有搜索缓冲区
    for (int i = 0; i < 7; i++)
    {
        search_data_buffers[i].clear();
        search_time_buffers[i].clear();
    }

    std::fill(Num.begin(), Num.end(), 0);


    std::cout << "[Tracker] 重置追踪器" << std::endl;
}