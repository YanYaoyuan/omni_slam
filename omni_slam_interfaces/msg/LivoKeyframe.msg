# Pose is T_header_tracking; cloud is expressed in tracking_frame at header.stamp.
# id is scoped to one mapping session. image_stamp is the actual camera time.
std_msgs/Header header
uint64 id
string tracking_frame
geometry_msgs/Pose pose
sensor_msgs/PointCloud2 cloud
builtin_interfaces/Time image_stamp
bool visual_update
