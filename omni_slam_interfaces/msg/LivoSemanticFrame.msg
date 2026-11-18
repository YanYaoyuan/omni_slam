# Atomic input for an asynchronous semantic consumer. Never feed estimates back.
LivoKeyframe keyframe
# Unannotated rectified image at image.header.stamp (not the LiDAR epoch).
sensor_msgs/Image image
sensor_msgs/CameraInfo camera_info
# T_camera(image_stamp)_tracking(keyframe_stamp), including frontend time compensation.
geometry_msgs/Transform camera_from_tracking
