#ifndef VISIONPILOT_VEHICLE_INTERFACE_HPP
#define VISIONPILOT_VEHICLE_INTERFACE_HPP

#include <vector>

class VehicleInterface
{
public:
    VehicleInterface() = default;
    virtual ~VehicleInterface() = default;

    // Read vehicle speed via CAN frame
    virtual double read() = 0;

    // Send steering and acceleration via CAN frame
    virtual void write(double steering, double acceleration) = 0;

    // Publish fused lane-center path. Default no-op so CAN/file backends
    // are unchanged. Coefficients are y = a x² + b x + c in base_link
    // (x forward [m], y left [m]). Invalid/empty when valid is false.
    virtual void publish_lane_path(
        bool valid,
        float path_a,
        float path_b,
        float path_c,
        float path_x_max_m)
    {
        (void)valid;
        (void)path_a;
        (void)path_b;
        (void)path_c;
        (void)path_x_max_m;
    }

    // Native speed schedule (m/s). Default no-op for CAN/file backends.
    virtual void publish_speed_horizon(const std::vector<double>& speeds)
    {
        (void)speeds;
    }
};


#endif //VISIONPILOT_VEHICLE_INTERFACE_HPP
