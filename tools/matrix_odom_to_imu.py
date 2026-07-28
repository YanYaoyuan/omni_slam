#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu


def quat_to_matrix(q: Quaternion):
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def world_to_body(rot, vec):
    return (
        rot[0][0] * vec[0] + rot[1][0] * vec[1] + rot[2][0] * vec[2],
        rot[0][1] * vec[0] + rot[1][1] * vec[1] + rot[2][1] * vec[2],
        rot[0][2] * vec[0] + rot[1][2] * vec[1] + rot[2][2] * vec[2],
    )


class MatrixOdomToImu(Node):
    def __init__(self):
        super().__init__("matrix_odom_to_imu")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("imu_topic", "/livox/imu")
        self.declare_parameter("imu_frame_id", "livox_frame")

        odom_topic = self.get_parameter("odom_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        self.imu_frame_id = self.get_parameter("imu_frame_id").value

        self.last_stamp = None
        self.last_v_world = None
        self.pub = self.create_publisher(Imu, imu_topic, 100)
        self.sub = self.create_subscription(Odometry, odom_topic, self.on_odom, 100)
        self.get_logger().warn(
            "Synthetic IMU is for simulation bring-up only. Prefer a real simulated IMU topic when available."
        )

    def on_odom(self, odom: Odometry):
        stamp = odom.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        q = odom.pose.pose.orientation
        rot = quat_to_matrix(q)

        v_world = (
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        )
        acc_world = (0.0, 0.0, 0.0)
        if self.last_stamp is not None and self.last_v_world is not None:
            dt = t - self.last_stamp
            if dt > 1e-4:
                acc_world = tuple((v_world[i] - self.last_v_world[i]) / dt for i in range(3))

        self.last_stamp = t
        self.last_v_world = v_world

        specific_force_world = (acc_world[0], acc_world[1], acc_world[2] + 9.80665)
        acc_body = world_to_body(rot, specific_force_world)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = self.imu_frame_id
        imu.orientation = q
        imu.angular_velocity = odom.twist.twist.angular
        imu.linear_acceleration.x = acc_body[0]
        imu.linear_acceleration.y = acc_body[1]
        imu.linear_acceleration.z = acc_body[2]
        imu.orientation_covariance[0] = 0.01
        imu.orientation_covariance[4] = 0.01
        imu.orientation_covariance[8] = 0.01
        imu.angular_velocity_covariance[0] = 0.01
        imu.angular_velocity_covariance[4] = 0.01
        imu.angular_velocity_covariance[8] = 0.01
        imu.linear_acceleration_covariance[0] = 0.1
        imu.linear_acceleration_covariance[4] = 0.1
        imu.linear_acceleration_covariance[8] = 0.1
        self.pub.publish(imu)


def main():
    rclpy.init()
    node = MatrixOdomToImu()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
