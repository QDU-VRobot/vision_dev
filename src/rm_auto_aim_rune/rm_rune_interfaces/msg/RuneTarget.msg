# 能量机关检测完整结果
std_msgs/Header header

bool detected                           # 是否检测到能量机关

geometry_msgs/Pose center_pose          # 能量机关中心三维位姿 (相机坐标系, 单位: m)
float64 current_angle                   # 当前转角 (rad)
float64 angular_velocity                # 角速度 (rad/s)

rm_rune_interfaces/Rune[] runes         # 各扇叶信息
