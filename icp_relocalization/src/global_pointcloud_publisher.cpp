#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class GlobalPointcloudPublisher : public rclcpp::Node
{
public:
    GlobalPointcloudPublisher()
        : Node("omni_global_pointcloud_publisher")
    {
        input_topic_ = declare_parameter<std::string>(
            "input_topic", "/cloud_registered");
        output_topic_ = declare_parameter<std::string>(
            "output_topic", "/cloud_registered_global");
        target_frame_ = declare_parameter<std::string>(
            "target_frame", "omni_map");
        source_frame_ = declare_parameter<std::string>(
            "source_frame", "omni_odom");
        const auto transform_timeout_ms = declare_parameter<std::int64_t>(
            "transform_timeout_ms", 250);

        if (input_topic_.empty() || output_topic_.empty() ||
            target_frame_.empty() || source_frame_.empty())
        {
            throw std::invalid_argument(
                "pointcloud topics and frame parameters must not be empty");
        }
        if (input_topic_ == output_topic_)
        {
            throw std::invalid_argument(
                "input_topic and output_topic must be different");
        }
        if (target_frame_ == source_frame_)
        {
            throw std::invalid_argument(
                "target_frame and source_frame must be different");
        }
        if (transform_timeout_ms < 0 || transform_timeout_ms > 1000)
        {
            throw std::invalid_argument(
                "transform_timeout_ms must be in [0, 1000]");
        }
        transform_timeout_sec_ =
            static_cast<double>(transform_timeout_ms) / 1000.0;

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        const auto qos = rclcpp::SensorDataQoS();
        publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_, qos);
        subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            qos,
            std::bind(
                &GlobalPointcloudPublisher::pointcloudCallback,
                this,
                std::placeholders::_1));

        RCLCPP_INFO(
            get_logger(),
            "Converting pointcloud %s (%s) to %s (%s)",
            input_topic_.c_str(),
            source_frame_.c_str(),
            output_topic_.c_str(),
            target_frame_.c_str());
    }

private:
    void pointcloudCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
    {
        if (message->header.frame_id != source_frame_)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Rejecting %s with frame_id '%s'; expected '%s'",
                input_topic_.c_str(),
                message->header.frame_id.c_str(),
                source_frame_.c_str());
            return;
        }

        geometry_msgs::msg::TransformStamped target_from_source;
        try
        {
            // map->odom 与注册点云都由传感器帧驱动，点云可能比同时间戳 TF
            // 早到一个发布周期。短暂等待对应 TF，避免把正常的调度先后误判成
            // 数据丢失，进而让 SLAM Manager 错误退回 RELOCALIZING。
            target_from_source = tf_buffer_->lookupTransform(
                target_frame_,
                source_frame_,
                rclcpp::Time(message->header.stamp),
                rclcpp::Duration::from_seconds(transform_timeout_sec_));
        }
        catch (const tf2::TransformException &exception)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Waiting for transform %s <- %s at cloud stamp: %s",
                target_frame_.c_str(),
                source_frame_.c_str(),
                exception.what());
            return;
        }

        tf2::Transform transform;
        tf2::fromMsg(target_from_source.transform, transform);
        tf2::Quaternion rotation = transform.getRotation();
        if (rotation.length2() < 1.0e-12)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Rejecting invalid transform %s <- %s",
                target_frame_.c_str(),
                source_frame_.c_str());
            return;
        }
        rotation.normalize();
        transform.setRotation(rotation);

        sensor_msgs::msg::PointCloud2 output = *message;
        output.header.frame_id = target_frame_;

        try
        {
            sensor_msgs::PointCloud2Iterator<float> x(output, "x");
            sensor_msgs::PointCloud2Iterator<float> y(output, "y");
            sensor_msgs::PointCloud2Iterator<float> z(output, "z");

            for (; x != x.end(); ++x, ++y, ++z)
            {
                if (!std::isfinite(*x) ||
                    !std::isfinite(*y) ||
                    !std::isfinite(*z))
                {
                    continue;
                }

                const tf2::Vector3 source_point(*x, *y, *z);
                const tf2::Vector3 target_point = transform * source_point;
                *x = static_cast<float>(target_point.x());
                *y = static_cast<float>(target_point.y());
                *z = static_cast<float>(target_point.z());
            }
        }
        catch (const std::runtime_error &exception)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Cannot transform %s: %s",
                input_topic_.c_str(),
                exception.what());
            return;
        }

        publisher_->publish(output);
        RCLCPP_INFO_ONCE(
            get_logger(),
            "Publishing global pointcloud in frame %s",
            target_frame_.c_str());
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string target_frame_;
    std::string source_frame_;
    double transform_timeout_sec_{0.25};

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalPointcloudPublisher>());
    rclcpp::shutdown();
    return 0;
}
