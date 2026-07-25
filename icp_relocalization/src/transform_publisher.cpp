#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <rclcpp/rclcpp.hpp>

class TransformPublisherNode : public rclcpp::Node
{
public:
    TransformPublisherNode()
        : Node("transform_publisher_node")
    {
        odom_frame_id_ = this->declare_parameter<std::string>(
            "odom_frame_id", "odom");
        map_frame_id_ = this->declare_parameter<std::string>(
            "map_frame_id", "map");
        icp_result_topic_ = this->declare_parameter<std::string>(
            "icp_result_topic", "/icp_result");

        if (odom_frame_id_.empty() || map_frame_id_.empty())
        {
            throw std::invalid_argument(
                "map_frame_id and odom_frame_id must not be empty");
        }
        if (odom_frame_id_ == map_frame_id_)
        {
            throw std::invalid_argument(
                "map_frame_id and odom_frame_id must be different");
        }
        if (icp_result_topic_.empty())
        {
            throw std::invalid_argument("icp_result_topic must not be empty");
        }

        const auto qos =
            rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
        subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            icp_result_topic_,
            qos,
            std::bind(
                &TransformPublisherNode::callback,
                this,
                std::placeholders::_1));
        broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        RCLCPP_INFO(
            this->get_logger(),
            "Waiting for the first valid ICP pose on %s to publish %s -> %s",
            icp_result_topic_.c_str(),
            map_frame_id_.c_str(),
            odom_frame_id_.c_str());
    }

private:
    void callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        if (initialized_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Ignoring a later ICP result because %s -> %s is already initialized",
                map_frame_id_.c_str(),
                odom_frame_id_.c_str());
            return;
        }

        if (msg->header.frame_id != map_frame_id_)
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "Rejecting %s with frame_id '%s'; expected '%s'",
                icp_result_topic_.c_str(),
                msg->header.frame_id.c_str(),
                map_frame_id_.c_str());
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
            RCLCPP_ERROR(
                this->get_logger(),
                "Rejecting %s because its pose contains a non-finite value",
                icp_result_topic_.c_str());
            return;
        }

        tf2::Quaternion rotation(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w);
        if (rotation.length2() < 1.0e-12)
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "Rejecting %s because its quaternion has near-zero length",
                icp_result_topic_.c_str());
            return;
        }
        rotation.normalize();

        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = msg->header.stamp;
        if (transform.header.stamp.sec == 0 &&
            transform.header.stamp.nanosec == 0)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "%s has a zero timestamp; using the node clock instead",
                icp_result_topic_.c_str());
            transform.header.stamp = this->now();
        }
        transform.header.frame_id = map_frame_id_;
        transform.child_frame_id = odom_frame_id_;
        transform.transform.translation.x = position.x;
        transform.transform.translation.y = position.y;
        transform.transform.translation.z = position.z;
        transform.transform.rotation = tf2::toMsg(rotation);

        broadcaster_->sendTransform(transform);
        initialized_ = true;

        RCLCPP_INFO(
            this->get_logger(),
            "Published static transform %s -> %s from the first valid ICP result",
            map_frame_id_.c_str(),
            odom_frame_id_.c_str());
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr subscription_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster_;
    std::string odom_frame_id_;
    std::string map_frame_id_;
    std::string icp_result_topic_;
    bool initialized_ = false;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TransformPublisherNode>());
    rclcpp::shutdown();
    return 0;
}
