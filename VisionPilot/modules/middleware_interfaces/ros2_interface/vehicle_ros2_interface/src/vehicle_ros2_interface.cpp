#include <cmath>
#include <string>
#include <vector>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vehicle_ros2_interface/vehicle_ros2_interface.hpp>

// ── VehicleRos2Node ───────────────────────────────────────────────────────────

VehicleRos2Interface::VehicleRos2Node::VehicleRos2Node(
    std::string vehicle_speed_topic, std::string vehicle_steering_topic,
    std::string vehicle_acceleration_topic,
    std::function<void(double)> on_speed) : rclcpp::Node("VehicleRos2Node")
{
    // Best-effort depth-1 QoS — we only need the freshest value.
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

    sub_ = create_subscription<std_msgs::msg::Float64>(
        vehicle_speed_topic, qos,
        [on_speed](const std_msgs::msg::Float64::SharedPtr msg)
        {
            on_speed(msg->data);
        });

    // Reliable depth-1 QoS for commands — must not be silently dropped.
    auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();

    steering_pub_ = create_publisher<std_msgs::msg::Float64>(vehicle_steering_topic, cmd_qos);
    throttle_pub_ = create_publisher<std_msgs::msg::Float64>(vehicle_acceleration_topic, cmd_qos);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/vehicle/lane_path", cmd_qos);
    speed_horizon_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "/vehicle/speed_horizon", cmd_qos);

    RCLCPP_INFO(get_logger(), "VehicleRos2Interface ready");
    RCLCPP_INFO(get_logger(), "  sub  /vehicle/speed");
    RCLCPP_INFO(get_logger(), "  pub  /vehicle/steering_cmd");
    RCLCPP_INFO(get_logger(), "  pub  /vehicle/throttle_cmd");
    RCLCPP_INFO(get_logger(), "  pub  /vehicle/lane_path");
    RCLCPP_INFO(get_logger(), "  pub  /vehicle/speed_horizon");
}

// ── VehicleRos2Interface ──────────────────────────────────────────────────────

VehicleRos2Interface::VehicleRos2Interface(std::string vehicle_speed_topic, std::string vehicle_steering_topic,
                                           std::string vehicle_acceleration_topic)
{
    node_ = std::make_shared < VehicleRos2Node > (vehicle_speed_topic,
        vehicle_steering_topic,
        vehicle_acceleration_topic,
        [this](double speed) { speed_.store(speed, std::memory_order_relaxed); });

    executor_.add_node(node_);
    spin_thread_ = std::thread([this]() { executor_.spin(); });
}

VehicleRos2Interface::~VehicleRos2Interface()
{
    executor_.cancel();
    if (spin_thread_.joinable()) spin_thread_.join();
}

// ── VehicleInterface implementation ──────────────────────────────────────────

double VehicleRos2Interface::read()
{
    return speed_.load(std::memory_order_relaxed);
}

void VehicleRos2Interface::write(const double steering, const double acceleration)
{
    std_msgs::msg::Float64 steer_msg;
    steer_msg.data = steering;
    node_->steering_pub_->publish(steer_msg);

    std_msgs::msg::Float64 throttle_msg;
    throttle_msg.data = acceleration;
    node_->throttle_pub_->publish(throttle_msg);
}

void VehicleRos2Interface::publish_lane_path(
    const bool valid,
    const float path_a,
    const float path_b,
    const float path_c,
    const float path_x_max_m)
{
    nav_msgs::msg::Path path;
    path.header.frame_id = "base_link";
    path.header.stamp = node_->now();
    if (valid)
    {
        float x_max = path_x_max_m;
        if (x_max < 10.0f) x_max = 40.0f;
        if (x_max > 60.0f) x_max = 60.0f;
        constexpr float kSpacing = 1.0f;
        for (float x = 0.0f; x <= x_max + 1e-3f; x += kSpacing)
        {
            const float y = path_a * x * x + path_b * x + path_c;
            const float yaw = std::atan(2.0f * path_a * x + path_b);
            const float half = 0.5f * yaw;
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position.x = x;
            pose.pose.position.y = y;
            pose.pose.orientation.z = std::sin(half);
            pose.pose.orientation.w = std::cos(half);
            path.poses.push_back(pose);
        }
    }
    node_->path_pub_->publish(path);
}

void VehicleRos2Interface::publish_speed_horizon(const std::vector<double>& speeds)
{
    std_msgs::msg::Float32MultiArray msg;
    msg.data.reserve(speeds.size());
    for (double v : speeds)
    {
        msg.data.push_back(static_cast<float>(v));
    }
    node_->speed_horizon_pub_->publish(msg);
}
