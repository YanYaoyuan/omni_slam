#include <array>
#include <cmath>
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
        source_frame_ = this->declare_parameter<std::string>(
            "source_frame", "lio_odom");
        child_frame_ = this->declare_parameter<std::string>(
            "child_frame", "livox_frame");
        initial_translation_tolerance_ = this->declare_parameter<double>(
            "initial_translation_tolerance", 0.5);
        initial_rotation_tolerance_ = this->declare_parameter<double>(
            "initial_rotation_tolerance", 0.35);

        if (input_topic_.empty() ||
            output_topic_.empty() ||
            target_frame_.empty() ||
            source_frame_.empty() ||
            child_frame_.empty())
        {
            throw std::invalid_argument(
                "topic and frame parameters must not be empty");
        }
        if (input_topic_ == output_topic_)
        {
            throw std::invalid_argument(
                "input_topic and output_topic must be different");
        }
        if (target_frame_ == source_frame_ ||
            target_frame_ == child_frame_ ||
            source_frame_ == child_frame_)
        {
            throw std::invalid_argument(
                "target_frame, source_frame, and child_frame must be different");
        }
        if (!std::isfinite(initial_translation_tolerance_) ||
            !std::isfinite(initial_rotation_tolerance_) ||
            initial_translation_tolerance_ < 0.0 ||
            initial_rotation_tolerance_ < 0.0)
        {
            throw std::invalid_argument(
                "initial pose tolerances must be finite and non-negative");
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
            "Converting %s (%s -> %s) to %s (%s -> %s)",
            input_topic_.c_str(),
            source_frame_.c_str(),
            child_frame_.c_str(),
            output_topic_.c_str(),
            target_frame_.c_str(),
            child_frame_.c_str());
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
        if (msg->header.frame_id != source_frame_ ||
            msg->child_frame_id != child_frame_)
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Rejecting %s with frames '%s' -> '%s'; expected '%s' -> '%s'",
                input_topic_.c_str(),
                msg->header.frame_id.c_str(),
                msg->child_frame_id.c_str(),
                source_frame_.c_str(),
                child_frame_.c_str());
            return;
        }

        const auto &position = msg->pose.pose.position;
        const auto &orientation = msg->pose.pose.orientation;
        if (!std::isfinite(position.x) ||
            !std::isfinite(position.y) ||
            !std::isfinite(position.z) ||
            !std::isfinite(orientation.x) ||
            !std::isfinite(orientation.y) ||
            !std::isfinite(orientation.z) ||
            !std::isfinite(orientation.w))
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Rejecting %s because its pose contains a non-finite value",
                input_topic_.c_str());
            return;
        }

        tf2::Quaternion source_rotation(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w);
        if (source_rotation.length2() < 1.0e-12)
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Rejecting %s because its quaternion has near-zero length",
                input_topic_.c_str());
            return;
        }
        source_rotation.normalize();

        if (!initial_local_pose_checked_)
        {
            initial_local_pose_checked_ = true;
            const double translation =
                std::sqrt(
                    position.x * position.x +
                    position.y * position.y +
                    position.z * position.z);
            const double rotation = source_rotation.getAngleShortestPath();
            if (translation > initial_translation_tolerance_ ||
                rotation > initial_rotation_tolerance_)
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "The first local pose is not close to identity "
                    "(translation %.3f m, rotation %.3f rad). "
                    "The current ICP-as-map-to-odom initialization assumes an "
                    "approximately identity first local pose.",
                    translation,
                    rotation);
            }
            else
            {
                RCLCPP_INFO(
                    this->get_logger(),
                    "First local pose identity check passed "
                    "(translation %.3f m, rotation %.3f rad)",
                    translation,
                    rotation);
            }
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
        source_pose.pose.orientation = tf2::toMsg(source_rotation);

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
    std::string source_frame_;
    std::string child_frame_;
    double initial_translation_tolerance_;
    double initial_rotation_tolerance_;
    bool initial_local_pose_checked_ = false;

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
