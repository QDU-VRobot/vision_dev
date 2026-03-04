# 单片能量机关扇叶信息
# Types
uint8 UNKNOWN=0
uint8 STRUCK=1
uint8 UNSTRUCK=2
uint8 PENDING_STRUCK=3

uint8 rune_type                       # 扇叶类型
geometry_msgs/Pose pose               # 扇叶三维位姿 (相机坐标系, 单位: m)
geometry_msgs/Vector3 velocity        # 扇叶速度 (m/s)
float64 angle_offset                  # 相对能量机关中心的角度偏移 (rad)
int32 drop_frame_count                # 掉帧计数
