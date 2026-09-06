#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include "fast_lio/extent_safe_voxel_filter.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/transform_datatypes.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "icp_relocalization/relocalization_convergence.hpp"
#ifdef USE_LIVOX
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

class ICPNode : public rclcpp::Node
{
public:
    ICPNode()
        : Node("omni_icp_relocalization")
    {
        this->declare_parameter("initial_x", 0.0);
        this->declare_parameter("initial_y", 0.0);
        this->declare_parameter("initial_z", 0.0);
        this->declare_parameter("initial_a", 0.0);
        this->declare_parameter("solver_max_iter", 75);
        this->declare_parameter("max_correspondence_distance", 0.1);
        this->declare_parameter("RANSAC_outlier_rejection_threshold", 1.0);
        this->declare_parameter("map_path", "");
        this->declare_parameter("map_frame_id", "map");
        this->declare_parameter("sensor_frame_id", "lidar");
        this->declare_parameter("fitness_score_thre", 0.0);
        this->declare_parameter("icp_fitness_topic", "icp_fitness");
        this->declare_parameter("map_voxel_leaf_size", 0.1);
        this->declare_parameter("cloud_voxel_leaf_size", 0.1);
        this->declare_parameter("converged_count_thre", 20);
        this->declare_parameter("candidate_position_tolerance_m", 0.35);
        this->declare_parameter("candidate_angle_tolerance_rad", 0.174533);
        this->declare_parameter("pcl_type","livox");
        this->declare_parameter("livox_topic", "/livox/lidar");
        this->declare_parameter("pointcloud_topic", "/pointcloud2");

        this->get_parameter("initial_x", initial_x);
        this->get_parameter("initial_y", initial_y);
        this->get_parameter("initial_z", initial_z);
        this->get_parameter("initial_a", initial_a);
        this->get_parameter("solver_max_iter", solver_max_iter);
        this->get_parameter("max_correspondence_distance", max_correspondence_distance);
        this->get_parameter("RANSAC_outlier_rejection_threshold", RANSAC_outlier_rejection_threshold);
        this->get_parameter("map_path", map_path);
        this->get_parameter("map_frame_id", map_frame);
        this->get_parameter("sensor_frame_id", sensor_frame);
        this->get_parameter("fitness_score_thre", fitness_score_thre);
        this->get_parameter("icp_fitness_topic", icp_fitness_topic);
        this->get_parameter("map_voxel_leaf_size", map_voxel_leaf_size);
        this->get_parameter("cloud_voxel_leaf_size", cloud_voxel_leaf_size);
        this->get_parameter("converged_count_thre", converged_count_thre);
        this->get_parameter(
            "candidate_position_tolerance_m", candidate_position_tolerance_m);
        this->get_parameter(
            "candidate_angle_tolerance_rad", candidate_angle_tolerance_rad);
        this->get_parameter("pcl_type", pcl_type);
        this->get_parameter("livox_topic", livox_topic);
        this->get_parameter("pointcloud_topic", pointcloud_topic);

        if (converged_count_thre <= 0)
        {
            throw std::invalid_argument("converged_count_thre must be positive");
        }
        convergence_gate_ =
            std::make_unique<icp_relocalization::RelocalizationConvergenceGate>(
                static_cast<std::size_t>(converged_count_thre),
                candidate_position_tolerance_m,
                candidate_angle_tolerance_rad);

        publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("icp_result", 10);
        fitness_pub_ = this->create_publisher<std_msgs::msg::Float64>(icp_fitness_topic, 10);
#ifdef USE_LIVOX
        if(pcl_type == "livox")
        {
            lvx_cloud_sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                livox_topic, 10, std::bind(&ICPNode::lvx_cloud_callback, this, std::placeholders::_1));
        }
        else
        {
            cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic, 10, std::bind(&ICPNode::cloud_callback, this, std::placeholders::_1));
        }
#else
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic, 10, std::bind(&ICPNode::cloud_callback, this, std::placeholders::_1));
#endif
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "initialpose", 10, std::bind(&ICPNode::pose_callback, this, std::placeholders::_1));
        map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("prior_map", 10);
        transformed_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("transformed_cloud", 10);
        
        // init guess
        initGuess = Eigen::Matrix4f::Identity();
        initGuess(0, 3) = initial_x;
        initGuess(1, 3) = initial_y;
        initGuess(2, 3) = initial_z;
        // You need to convert the quaternion to a rotation matrix and set it to the upper-left 3x3 part of the matrix
        tf2::Quaternion q;
        q.setRPY(0, 0, initial_a);
        tf2::Matrix3x3 rot_mat(q);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                initGuess(i, j) = rot_mat[i][j];
            }
        }
        RCLCPP_INFO(this->get_logger(), "Initial guess: \n x: %f, y: %f, z: %f, a: %f", initial_x, initial_y, initial_z, initial_a);
        RCLCPP_INFO(
            this->get_logger(),
            "Relocalization convergence gate: samples=%d position_tolerance=%.3f m angle_tolerance=%.3f rad",
            converged_count_thre, candidate_position_tolerance_m,
            candidate_angle_tolerance_rad);
        // Load the target point cloud from a PCD file
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path, *target_cloud_) == -1)
        {
            RCLCPP_FATAL(this->get_logger(), "Couldn't read map file: %s", map_path.c_str());
            throw std::runtime_error("failed to load ICP map");
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %d data points from target.pcd", target_cloud_->width * target_cloud_->height);

        // 普通 VoxelGrid 的全局体素索引使用 32 位乘积，少量远距离离群点就会
        // 让它放弃过滤。使用与 FAST-LIO 相同的 64 位哈希体素实现，保证 ICP
        // 和持续定位消费完全一致的先验地图分辨率。
        const auto filter_stats = fast_lio::pointcloud::extentSafeVoxelFilter(
            *target_cloud_, map_voxel_leaf_size, target_cloud_.get());
        if (target_cloud_->empty())
        {
            RCLCPP_FATAL(this->get_logger(), "Map is empty after extent-safe voxel filtering: %s", map_path.c_str());
            throw std::runtime_error("filtered ICP map is empty");
        }
        RCLCPP_INFO(
            this->get_logger(),
            "Target map ready: raw_points=%zu filtered_points=%zu rejected_non_finite=%zu rejected_index_overflow=%zu",
            filter_stats.input_points, filter_stats.output_points,
            filter_stats.rejected_non_finite, filter_stats.rejected_index_overflow);
        // Publish the downsampled target cloud

        pcl::toROSMsg(*target_cloud_, target_cloud_msg);
        target_cloud_msg.header.stamp = this->now();
        target_cloud_msg.header.frame_id = map_frame;
        map_pub_->publish(target_cloud_msg);
    }

private:
    void filter_input_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        filtered->reserve(cloud->size());

        constexpr float min_range_squared = 0.3f * 0.3f;
        constexpr float max_range_squared = 30.0f * 30.0f;
        for (const auto & point : cloud->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            {
                continue;
            }
            const float range_squared =
                point.x * point.x + point.y * point.y + point.z * point.z;
            if (range_squared < min_range_squared || range_squared > max_range_squared)
            {
                continue;
            }
            filtered->push_back(point);
        }

        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = true;
        if (filtered->size() != cloud->size())
        {
            RCLCPP_DEBUG(
                this->get_logger(), "Filtered %zu invalid/out-of-range points from ICP input",
                cloud->size() - filtered->size());
        }
        cloud.swap(filtered);
    }

    using IcpSolver =
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>;

    /**
     * @brief 评估一帧 ICP 结果，并在形成稳定候选簇后给出最终初始位姿。
     *
     * fitness 只说明点到点残差，不能排除重复结构导致的跳解。因此这里还要求
     * 连续候选在 SE(3) 中相互一致。质量合格的候选会成为下一帧 ICP 初值，
     * 避免 PointCloud2 分支持续从用户给定的旧初值重新搜索。
     */
    bool evaluate_candidate(
        IcpSolver & icp,
        const double fitness_score,
        Eigen::Matrix4f & stable_transformation)
    {
        const bool quality_accepted =
            icp.hasConverged() && std::isfinite(fitness_score) &&
            fitness_score < fitness_score_thre;
        const Eigen::Matrix4f candidate = quality_accepted ?
            icp.getFinalTransformation() : initGuess;
        const auto decision =
            convergence_gate_->Observe(quality_accepted, candidate);

        if (!quality_accepted)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "ICP candidate rejected: converged=%s fitness=%.6f threshold=%.6f; stable evidence cleared",
                icp.hasConverged() ? "true" : "false", fitness_score,
                fitness_score_thre);
            return false;
        }

        if (decision == icp_relocalization::ConvergenceDecision::kRejected)
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "ICP returned a non-finite or non-rigid transformation; candidate discarded and stable evidence cleared");
            return false;
        }

        // 仅用通过基础质量门的有限刚体变换更新搜索种子；高残差或非法结果
        // 不得污染下一帧初值。
        initGuess = candidate;
        const double yaw = std::atan2(candidate(1, 0), candidate(0, 0));
        if (decision == icp_relocalization::ConvergenceDecision::kClusterRestarted)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "ICP candidate changed pose cluster; convergence evidence restarted at X=%.3f Y=%.3f Z=%.3f YAW=%.3f rad, fitness=%.6f",
                candidate(0, 3), candidate(1, 3), candidate(2, 3), yaw,
                fitness_score);
            return false;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Relocalization stable candidate [%zu/%zu]: X=%.3f Y=%.3f Z=%.3f YAW=%.3f rad (%.1f deg), fitness=%.6f",
            convergence_gate_->sample_count(),
            convergence_gate_->required_samples(),
            candidate(0, 3), candidate(1, 3), candidate(2, 3), yaw,
            yaw * 180.0 / M_PI, fitness_score);
        if (decision != icp_relocalization::ConvergenceDecision::kReady)
        {
            return false;
        }

        // 发布稳定簇的均值，而不是最后一帧的瞬时解，降低初始化 TF 抖动。
        stable_transformation = convergence_gate_->Estimate();
        return true;
    }

    void publish_transformed_cloud(
        pcl::PointCloud<pcl::PointXYZ>::Ptr & input_cloud,
        const Eigen::Matrix4f & transformation)
    {
        pcl::transformPointCloud(
            *input_cloud, *input_cloud, transformation);
        sensor_msgs::msg::PointCloud2 transformed_cloud_msg;
        pcl::toROSMsg(*input_cloud, transformed_cloud_msg);
        transformed_cloud_msg.header.stamp = this->now();
        transformed_cloud_msg.header.frame_id = map_frame;
        transformed_cloud_pub_->publish(transformed_cloud_msg);

        target_cloud_msg.header.stamp = this->now();
        map_pub_->publish(target_cloud_msg);
    }

    template<typename StampT>
    void publish_relocalization_result(
        const StampT & source_stamp,
        const Eigen::Matrix4f & transformation,
        const double fitness_score)
    {
        // ICP 对齐结果表示 T_map_sensor，由 TF Manager 统一换算并维护
        // T_map_odom；ICP 节点本身不争用 TF 发布权。
        nav_msgs::msg::Odometry pose_msg;
        pose_msg.header.stamp = source_stamp;
        if (pose_msg.header.stamp.sec == 0 &&
            pose_msg.header.stamp.nanosec == 0)
        {
            pose_msg.header.stamp = this->now();
        }
        pose_msg.header.frame_id = map_frame;
        pose_msg.child_frame_id = sensor_frame;
        pose_msg.pose.pose.position.x = transformation(0, 3);
        pose_msg.pose.pose.position.y = transformation(1, 3);
        pose_msg.pose.pose.position.z = transformation(2, 3);
        const Eigen::Quaternionf orientation(
            transformation.block<3, 3>(0, 0));
        pose_msg.pose.pose.orientation.x = orientation.x();
        pose_msg.pose.pose.orientation.y = orientation.y();
        pose_msg.pose.pose.orientation.z = orientation.z();
        pose_msg.pose.pose.orientation.w = orientation.w();
        publisher_->publish(pose_msg);

        const double yaw = std::atan2(
            transformation(1, 0), transformation(0, 0));
        RCLCPP_INFO(
            this->get_logger(),
            "RELOCALIZATION SUCCEEDED: published /icp_result, X=%.3f Y=%.3f Z=%.3f YAW=%.3f rad (%.1f deg), fitness=%.6f",
            transformation(0, 3), transformation(1, 3), transformation(2, 3),
            yaw, yaw * 180.0 / M_PI, fitness_score);
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert the incoming point cloud to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *input_cloud);
        filter_input_cloud(input_cloud);

        if (input_cloud->empty())
        {
            convergence_gate_->Reset();
            RCLCPP_WARN(this->get_logger(), "ICP input cloud is empty after filtering");
            return;
        }

        // Downsample the input cloud
        pcl::VoxelGrid<pcl::PointXYZ> sor_scan;
        sor_scan.setInputCloud(input_cloud);
        sor_scan.setLeafSize(cloud_voxel_leaf_size, cloud_voxel_leaf_size, cloud_voxel_leaf_size);
        sor_scan.filter(*input_cloud);
        RCLCPP_INFO(this->get_logger(), "Downsampled input cloud to %d data points", input_cloud->width * input_cloud->height);

        // Rotate pcl alone x axis for 180 degree
        // Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
        // rotation(1, 1) = -1;
        // rotation(2, 2) = -1;
        // pcl::transformPointCloud(*input_cloud, *input_cloud, rotation);

        // Perform ICP alignment
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(input_cloud);
        icp.setInputTarget(target_cloud_);
        icp.setMaximumIterations(solver_max_iter);
        // icp.setTransformationEpsilon(1e-8);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);
        icp.setRANSACOutlierRejectionThreshold(RANSAC_outlier_rejection_threshold);
        // icp.setRANSACIterations(100);
        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        icp.align(final_cloud, initGuess);

        // Get fitness score
        double fitness_score = icp.getFitnessScore(max_correspondence_distance);
        RCLCPP_INFO(this->get_logger(), "ICP fitness score: %f", fitness_score);
        std_msgs::msg::Float64 fitness_msg;
        fitness_msg.data = fitness_score;
        fitness_pub_->publish(fitness_msg);

        Eigen::Matrix4f stable_transformation = Eigen::Matrix4f::Identity();
        const bool ready = evaluate_candidate(
            icp, fitness_score, stable_transformation);
        publish_transformed_cloud(
            input_cloud, ready ? stable_transformation : initGuess);
        if (ready)
        {
            publish_relocalization_result(
                msg->header.stamp, stable_transformation, fitness_score);
            rclcpp::shutdown();
        }
    }

#ifdef USE_LIVOX
    void lvx_cloud_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        // Convert the incoming point cloud to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (int i = 0; i < msg->point_num; i++)
        {
            pcl::PointXYZ point;
            point.x = msg->points[i].x;
            point.y = msg->points[i].y;
            point.z = msg->points[i].z;
            input_cloud->push_back(point);
        }
        input_cloud->width = input_cloud->size();
        input_cloud->height = 1;
        filter_input_cloud(input_cloud);

        if (input_cloud->empty())
        {
            convergence_gate_->Reset();
            RCLCPP_WARN(this->get_logger(), "ICP input cloud is empty after filtering");
            return;
        }

        // Downsample the input cloud
        pcl::VoxelGrid<pcl::PointXYZ> sor_scan;
        sor_scan.setInputCloud(input_cloud);
        sor_scan.setLeafSize(cloud_voxel_leaf_size, cloud_voxel_leaf_size, cloud_voxel_leaf_size);
        sor_scan.filter(*input_cloud);
        RCLCPP_INFO(this->get_logger(), "Downsampled input cloud to %d data points", input_cloud->width * input_cloud->height);

        // Rotate pcl alone x axis for 180 degree
        // Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
        // rotation(1, 1) = -1;
        // rotation(2, 2) = -1;
        // pcl::transformPointCloud(*input_cloud, *input_cloud, rotation);

        // Perform ICP alignment
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(input_cloud);
        icp.setInputTarget(target_cloud_);
        icp.setMaximumIterations(solver_max_iter);
        // icp.setTransformationEpsilon(1e-8);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);
        icp.setRANSACOutlierRejectionThreshold(RANSAC_outlier_rejection_threshold);
        // icp.setRANSACIterations(100);
        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        icp.align(final_cloud, initGuess);

        // Get fitness score
        double fitness_score = icp.getFitnessScore(max_correspondence_distance);
        RCLCPP_INFO(this->get_logger(), "ICP fitness score: %f", fitness_score);
        std_msgs::msg::Float64 fitness_msg;
        fitness_msg.data = fitness_score;
        fitness_pub_->publish(fitness_msg);

        Eigen::Matrix4f stable_transformation = Eigen::Matrix4f::Identity();
        const bool ready = evaluate_candidate(
            icp, fitness_score, stable_transformation);

        publish_transformed_cloud(
            input_cloud, ready ? stable_transformation : initGuess);
        if (ready)
        {
            publish_relocalization_result(
                msg->header.stamp, stable_transformation, fitness_score);
            rclcpp::shutdown();
        }
    }
#endif

    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        // Convert the incoming pose to an Eigen matrix
        initGuess = Eigen::Matrix4f::Identity();
        initGuess(0, 3) = msg->pose.pose.position.x;
        initGuess(1, 3) = msg->pose.pose.position.y;
        initGuess(2, 3) = msg->pose.pose.position.z;
        // You need to convert the quaternion to a rotation matrix and set it to the upper-left 3x3 part of the matrix
        tf2::Quaternion q;
        tf2::fromMsg(msg->pose.pose.orientation, q);
        tf2::Matrix3x3 rot_mat(q);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                initGuess(i, j) = rot_mat[i][j];
            }
        }
        // 操作者更新初值后，旧候选簇已不属于同一次搜索，必须清空。
        convergence_gate_->Reset();
        double r,p,yaw;
        rot_mat.getRPY(r, p, yaw);
        RCLCPP_INFO(this->get_logger(), "Initial guess: \n x: %f, y: %f, z: %f, a: %f", msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, yaw);
    }

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr fitness_pub_;
    std::string icp_fitness_topic;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
#ifdef USE_LIVOX
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr lvx_cloud_sub_;
#endif
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr transformed_cloud_pub_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_{new pcl::PointCloud<pcl::PointXYZ>};

    Eigen::Matrix4f initGuess;
    double initial_x, initial_y, initial_z, initial_a;
    int solver_max_iter;
    double max_correspondence_distance, RANSAC_outlier_rejection_threshold;
    std::string map_path, map_frame, sensor_frame;
    double fitness_score_thre;
    double map_voxel_leaf_size, cloud_voxel_leaf_size;
    sensor_msgs::msg::PointCloud2 target_cloud_msg;
    int converged_count_thre;
    double candidate_position_tolerance_m;
    double candidate_angle_tolerance_rad;
    std::unique_ptr<icp_relocalization::RelocalizationConvergenceGate>
        convergence_gate_;
    std::string pcl_type;
    std::string livox_topic;
    std::string pointcloud_topic;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ICPNode>());
    rclcpp::shutdown();
    return 0;
}
