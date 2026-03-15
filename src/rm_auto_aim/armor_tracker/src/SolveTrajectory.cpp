#include "armor_tracker/SolveTrajectory.hpp"
#include <cmath>

namespace rm_auto_aim
{
SolveTrajectory::SolveTrajectory(const float& /* bias_time */) {}
float SolveTrajectory::SolvePitch(float x, float y, float z) { return 0.0f; }
float SolveTrajectory::SolveYaw(float x, float y,float z) { return 0.0f; }
void SolveTrajectory::PredictArmorPosition(const auto_aim_interfaces::msg::Target::SharedPtr&, float) {}
void SolveTrajectory::AutoSolveTrajectory(float& pitch, float& yaw, bool& is_fire, float& aim_x, float& aim_y, float& aim_z, const auto_aim_interfaces::msg::Target::SharedPtr msg) 
{
    pitch = 0; yaw = 0; is_fire = false; aim_x = 0; aim_y = 0; aim_z = 0;
}
} // namespace rm_auto_aim