#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class GlobalOdometryPublisher : public rclcpp::Node
{
public:
    GlobalOdometryPublisher()
        : Node("global_odometry_publisher")
    {
        input_topic_ = this->declare_parameter<std::string>(
            "input_topic", "/state_estimation");
        output_topic_ = this->declare_parameter<std::string>(
            "output_topic", "/state_estimation_global");
        target_frame_ = this->declare_parameter<std::string>(
            "target_frame", "lio_map");

        if (input_topic_.empty() || output_topic_.empty() || target_frame_.empty())
        {
            throw std::invalid_argument(
                "input_topic, output_topic, and target_frame must not be empty");
        }
        if (input_topic_ == output_topic_)
        {
            throw std::invalid_argument(
                "input_topic and output_topic must be different");
        }

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        const auto qos = rclcpp::QoS(rclcpp::KeepLast(20)).reliable();
        global_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            output_topic_, qos);
        local_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            input_topic_,
            qos,
            std::bind(
                &GlobalOdometryPublisher::odometryCallback,
                this,
                std::placeholders::_1));

        RCLCPP_INFO(
            this->get_logger(),
            "Converting odometry from %s to %s in frame %s",
            input_topic_.c_str(),
            output_topic_.c_str(),
            target_frame_.c_str());
    }

private:
    static std::array<double, 36> transformPoseCovariance(
        const std::array<double, 36> &covariance,
        const geometry_msgs::msg::Quaternion &rotation_msg)
    {
        tf2::Quaternion quaternion;
        tf2::fromMsg(rotation_msg, quaternion);
        quaternion.normalize();
        const tf2::Matrix3x3 rotation(quaternion);

        // Pose covariance order is [x, y, z, rotation_x, rotation_y, rotation_z].
        // Changing the parent coordinate frame rotates both 3D error blocks.
        double transform[6][6] = {};
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                transform[row][column] = rotation[row][column];
                transform[row + 3][column + 3] = rotation[row][column];
            }
        }

        std::array<double, 36> transformed_covariance{};
        for (std::size_t row = 0; row < 6; ++row)
        {
            for (std::size_t column = 0; column < 6; ++column)
            {
                double value = 0.0;
                for (std::size_t left = 0; left < 6; ++left)
                {
                    for (std::size_t right = 0; right < 6; ++right)
                    {
                        value +=
                            transform[row][left] *
                            covariance[left * 6 + right] *
                            transform[column][right];
                    }
                }
                transformed_covariance[row * 6 + column] = value;
            }
        }
        return transformed_covariance;
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        if (msg->header.frame_id.empty())
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Ignoring odometry with an empty frame_id");
            return;
        }

        if (msg->header.frame_id == target_frame_)
        {
            nav_msgs::msg::Odometry output = *msg;
            output.header.frame_id = target_frame_;
            global_odom_pub_->publish(output);
            return;
        }

        geometry_msgs::msg::TransformStamped target_from_source;
        try
        {
            target_from_source = tf_buffer_->lookupTransform(
                target_frame_,
                msg->header.frame_id,
                rclcpp::Time(msg->header.stamp));
        }
        catch (const tf2::TransformException &exception)
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for transform %s <- %s: %s",
                target_frame_.c_str(),
                msg->header.frame_id.c_str(),
                exception.what());
            return;
        }

        geometry_msgs::msg::PoseStamped source_pose;
        source_pose.header = msg->header;
        source_pose.pose = msg->pose.pose;

        geometry_msgs::msg::PoseStamped global_pose;
        tf2::doTransform(source_pose, global_pose, target_from_source);

        nav_msgs::msg::Odometry output = *msg;
        output.header.stamp = msg->header.stamp;
        output.header.frame_id = target_frame_;
        output.child_frame_id = msg->child_frame_id;
        output.pose.pose = global_pose.pose;
        output.pose.covariance = transformPoseCovariance(
            msg->pose.covariance,
            target_from_source.transform.rotation);

        // Odometry twist is expressed in child_frame_id. Since the child frame
        // is unchanged, both twist and twist covariance remain unchanged.
        global_odom_pub_->publish(output);

        RCLCPP_INFO_ONCE(
            this->get_logger(),
            "Publishing global odometry in frame %s",
            target_frame_.c_str());
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string target_frame_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr local_odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr global_odom_pub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalOdometryPublisher>());
    rclcpp::shutdown();
    return 0;
}
