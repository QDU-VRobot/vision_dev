#include "armor_tracker/tracker.hpp"
#include <cmath>
#include <algorithm>

namespace rm_auto_aim
{
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff) : tracker_state(LOST), is_initialized_(false) {}
void Tracker::Init(const Armors::SharedPtr& armors_msg) {}
void Tracker::Update(const Armors::SharedPtr& armors_msg, float cur_yaw, float cur_pitch, float yaw_err, float pitch_err) {}
}  // namespace rm_auto_aim