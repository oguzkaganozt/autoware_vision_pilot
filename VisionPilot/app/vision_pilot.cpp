// VisionPilot — preprocess → inference → fusion → display
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <algorithm>
#include <tuple>

#include <config/vision_pilot_config.hpp>
#include <common/utils.hpp>
#include <engine/onnx_engine.hpp>
#include <vehicle_interface/vehicle_interface.hpp>
#include <vehicle_interface/can_interface.hpp>
#include <image_preprocessing/image_preprocessor.hpp>
#include <logging/logger.hpp>
#include <models/inference.hpp>
#include <planning/planning.hpp>
#include <visualization/visualization.hpp>
#include <debug/debug_draw.hpp>

#include "camera_interface/v4l2_camera_interface.hpp"
#include "camera_interface/file_interface.hpp"
#include "vehicle_interface/file_interface.hpp"

#if ENABLE_ROS2_INTERFACE
#include <rclcpp/rclcpp.hpp>
#include <camera_ros2_interface/camera_ros2_interface.hpp>
#include <vehicle_ros2_interface/vehicle_ros2_interface.hpp>
#endif

namespace ve = visionpilot::engine;
namespace vm = visionpilot::models;
namespace vd = visionpilot::debug;

// ── Close-range CIPO latch (object permanence) ─────────────────────────────
// Fusion reports free-road (150 m) whenever NEITHER network confirms a
// target — including when a stopped lead sits centimeters away, too close
// for either network to see (bumper fills the frame, bbox truncated).
// Executing that as free-road drives into the bumper (proven 2026-09-11:
// contact at ~0 m true gap while fusion reported 150 m / has_cipo=false).
//
// Policy: a confirmed-close track (< LATCH_DIST_M) that vanishes while ego
// is slow latches as a stopped lead. While latched the planner is fed the
// latched world model — NEVER raw flicker: untrusted confirms neither
// reach the planner nor rewrite memory. The reported gap is
// min(coast, HOLD_DIST_M): at/inside the IDM standstill the planner
// holds by itself (accel AND horizon come out consistent — clamping the
// instantaneous accel alone was proven useless, motion flows through the
// horizon). A lead that genuinely drives away stays confirmed while
// visible, earns a trusted streak and releases; a close object that
// vanishes without a trace while ego is stopped is ~always "too close
// to see". Holding (annoyance) beats grinding into a bumper (collision).
// Returns {has_cipo, cipo_dist_m, cipo_rel_vel_ms} for the planner contract.
struct CipoLatch
{
    static constexpr double LATCH_DIST_M   = 15.0;
    static constexpr double HOLD_DIST_M    = 2.0;   // <= IDM s0: planner holds by itself
    static constexpr double EGO_V_GATE_MPS = 3.0;   // fast cruise flicker unaffected
    static constexpr int    RELEASE_FRAMES = 5;     // single-frame ghosts must not release
    // A latch may only arm off a SOLID track: flickering ghosts (proven
    // 2026-09-11: 5 m phantom at spawn bricking the launch) confirm a
    // frame here and there but never N in a row, so they can neither arm
    // nor — via the release gate — disarm.
    static constexpr int    ARM_FRAMES     = 10;
    static constexpr double JUMP_GATE_M    = 4.0;
    // NOTE: earlier revisions clamped plan.acceleration while latched
    // (-1.0 rolling / +0.2 stopped). Removed: motion flows through the
    // speed horizon, which is integrated inside compute_plan from the
    // (already shaped) input — output clamping never touched it.

    double last_dist_m = 150.0;
    double odom_m      = 0.0;   // ego travel integral (self-contained)
    double latch_odom_m = 0.0;  // odom at last confirm
    int    confirm_streak = 0;
    bool   latched     = false;
    std::chrono::steady_clock::time_point last_t = std::chrono::steady_clock::now();

    std::tuple<bool, double, double> update(bool fused_cipo, double fused_dist_m,
                                            double fused_rel_vel_ms, double ego_v)
    {
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last_t).count();
        last_t = now;
        if (dt > 0.0 && dt < 1.0)
            odom_m += std::max(0.0, ego_v) * dt;

        if (fused_cipo)
        {
            ++confirm_streak;
            const bool trusted = (confirm_streak >= RELEASE_FRAMES);
            // Kinematic plausibility: at these speeds nothing moves 4 m
            // in one frame. A far jump on a short memory is a ghost, not
            // a departure (departures stay confirmed and walk out).
            const bool plausible = (last_dist_m >= LATCH_DIST_M) ||
                                   (std::fabs(fused_dist_m - last_dist_m) < JUMP_GATE_M);
            if (trusted && plausible)
            {
                // Trusted track only: single-frame ghosts must neither
                // release the latch nor rewrite its memory (proven: a
                // phantom 11 m confirm poisoned last_dist into free-road).
                last_dist_m  = fused_dist_m;
                latch_odom_m = odom_m;
            }
            if (trusted && plausible && latched)
            {
                latched = false;
                VP_INFO("[CIPO-latch] released — target re-confirmed at %.1f m", fused_dist_m);
            }
            if (latched)
            {
                // Hold the latched world model through flicker: untrusted
                // confirms must not reach the planner for even one frame
                // (a +1.5 blip schedules a launch the pipeline executes).
                return latched_output(ego_v);
            }
            return {true, fused_dist_m, fused_rel_vel_ms};
        }
        const bool was_solid = (confirm_streak >= ARM_FRAMES);
        confirm_streak = 0;
        if (was_solid && last_dist_m < LATCH_DIST_M && ego_v < EGO_V_GATE_MPS)
        {
            if (!latched)
            {
                latched = true;
                VP_INFO("[CIPO-latch] engaged — holding stopped lead at %.1f m", last_dist_m);
            }
        }
        if (latched)
            return latched_output(ego_v);
        return {false, fused_dist_m, fused_rel_vel_ms};
    }

    // World model while latched: the coasted gap, capped at the IDM
    // standstill so the planner holds by itself (accel and horizon agree;
    // no per-output clamping needed downstream).
    std::tuple<bool, double, double> latched_output(double ego_v) const
    {
        // Coast: the car may still be rolling when the track drops, so
        // freeze the *world* point, not the last number — subtract ego
        // travel since the confirm. Floors at 0.5 m (IDM gap floor).
        const double coasted = std::max(0.5, last_dist_m - (odom_m - latch_odom_m));
        return {true, std::min(coasted, HOLD_DIST_M), -ego_v};  // stopped lead
    }
};

int main(int argc, char** argv)
{
    Config cfg;
    try { cfg = load_vision_pilot_config(); }
    catch (const std::exception& e)
    {
        VP_ERROR("Config: %s", e.what());
        return 1;
    }

    // ── CLI flags ─────────────────────────────────────────────────────────────
    bool show_window = true;
    bool debug_viz = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--debug-viz") debug_viz = true;
        else if (arg == "--no-window") show_window = false;
    }

    std::shared_ptr<CameraInterface> camera_interface;
    std::shared_ptr<VehicleInterface> vehicle_interface;
#if ENABLE_ROS2_INTERFACE
    rclcpp::init(argc, argv);
    camera_interface = std::make_unique<CameraRos2Interface>(cfg.source.input_camera_topic);
    vehicle_interface = std::make_shared<VehicleRos2Interface>(cfg.vehicle_speed_topic,
                                                               cfg.vehicle_steering_topic,
                                                               cfg.vehicle_acceleration_topic);
#else
    if (cfg.source.mode == SourceMode::Video)
    {
        camera_interface = std::make_unique<camera_interface::FileInterface>(
            cfg.source.input_video, cfg.source.video_loop, cfg.source.video_realtime);
        vehicle_interface = std::make_shared<FileInterface>(cfg.source.input_vehicle_speed, cfg.source.video_loop);
    }
    else
    {
        camera_interface = std::make_unique<camera_interface::V4L2CameraInterface>(
            cfg.source.v4l2_device, static_cast<uint32_t>(cfg.source.v4l2_fps));
        vehicle_interface = std::make_shared<CanInterface>();
    }
#endif

    ImagePreprocessor preprocessor;
    ve::OnnxEngine engine(cfg.engine);
    vm::InferencePipeline pipeline(engine, cfg.inference);
    Planner planner(cfg.speed_limit, cfg.L);
    if (cfg.rrd_on) logging::Rerun::init(cfg.rrd_log);

    // ── Init visualization assets once based on mode ──────────────────────────
    if (debug_viz)
    {
        VP_INFO("[Viz] Debug mode — annotated telemetry overlay");
        vd::init_wheel_assets(cfg.wheel_dir);
        vd::init_homography();
    }
    else
    {
        VP_INFO("[Viz] Production mode — clean HUD");
        visualization::init_production_assets();
    }

    // ── Initialize camera interface ───────────────────────────────────────────

    if (!camera_interface || !camera_interface->is_device_open())
    {
        VP_ERROR("Cannot open frame source");
        return 1;
    }

    // ── Initialize display ────────────────────────────────────────────────────
    visualization::Visualization visualization({cfg.webrtc_on, cfg.webrtc_port, show_window});

    const cv::Size net_size(vm::AutoDrive::NET_W, vm::AutoDrive::NET_H);
    cv::Mat frame, warped, resized;
    bool h_resized_set = false;
    cv::Mat H = load_matrix("H.yaml", "H");
    while (true)
    {
        auto [ok, frame] = camera_interface->get_latest_frame();
        if (!ok || frame.empty())
        {
            if (cfg.source.mode == SourceMode::Video && !cfg.source.video_loop) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        preprocessor.preprocess(frame, warped, resized, net_size);
        cv::Size frame_size = frame.size();
        // One-time: tell the pipeline how to project AutoSteer/AutoSpeed outputs
        // back to world when those networks run on the plain-resized image.
        if (!h_resized_set)
        {
            pipeline.set_H_resized(H, frame_size);
            h_resized_set = true;
        }

        // ── Default frame no inference ────────────────────────────────────────────
        cv::Mat display_frame = resized;

        if (const auto r = pipeline.process(warped, resized))
        {
            // pipeline.latency().print();

            const double ego_v = vehicle_interface->read();
            const double cte = r->lateral.cte_m;
            const double epsi = r->lateral.yaw_rad;
            const double kappa = r->lateral.curvature;

            // has_cipo: tracker-based — true only when filter tracks a target
            // closer than D_MAX. cipo_raw_found alone must not gate the planner.
            // The latch keeps a confirmed-close track alive as a stopped lead
            // when both networks drop it at bumper range (object permanence).
            static constexpr double D_MAX = 150.0;
            static CipoLatch cipo_latch;
            const bool fused_cipo = r->cipo.valid && r->cipo.distance_m < D_MAX;
            const auto [has_cipo, cipo_dist, cipo_rel_vel] = cipo_latch.update(
                fused_cipo, r->cipo.distance_m, r->cipo.velocity_ms, ego_v);
            // CONTRACT: cipo_v is the ABSOLUTE lead speed (m/s). Fusion reports
            // RELATIVE velocity (negative = approaching), so convert at the
            // boundary. speed_limit doubles as the free-road absolute value.
            const double cipo_v = has_cipo ? std::max(0.0, ego_v + cipo_rel_vel)
                                         : cfg.speed_limit;

            const double raw_cte = r->lateral.path_valid
                                       ? static_cast<double>(r->lateral.raw_cte_m)
                                       : cte;
            // NOTE: no output clamping here on purpose. While latched the
            // planner input is already the hold model (<=2 m), so accel
            // AND horizon come out consistent. Clamping the instantaneous
            // accel alone was proven useless: motion flows through the
            // horizon, which is integrated inside compute_plan.
            const Plan plan = planner.compute_plan(
                cte, epsi, kappa, ego_v, has_cipo, cipo_v, cipo_dist);

            VP_INFO(
                "plan: tyre=%.4f rad  accel=%.3f m/s²  |  cte=%.2fm(raw=%.2fm) cte_dot=%+.2fm/s  epsi=%.3f epsi_dot=%+.3frad/s  kappa=%.4f  |  cipo=%s%s  dist=%.1f m  vel=%+.2f m/s",
                plan.steering.empty() ? 0.0 : plan.steering[1],
                plan.acceleration,
                cte,
                raw_cte,
                r->lateral.cte_rate_mps,
                epsi,
                r->lateral.yaw_rate_rps,
                kappa,
                has_cipo ? "true" : "false",
                cipo_latch.latched ? "[LATCH]" : "",
                cipo_dist,
                cipo_rel_vel);

            vehicle_interface->write(
                plan.steering.empty() ? 0.0 : plan.steering[1],
                plan.acceleration);
            vehicle_interface->publish_speed_horizon(plan.speed_horizon);
            vehicle_interface->publish_lane_path(
                r->lateral.path_valid,
                r->lateral.path_a,
                r->lateral.path_b,
                r->lateral.path_c,
                r->lateral.path_x_max_m);
            cv::Mat viz;  // output visualization image (empty when viz is off)
            if (cfg.visualization_on)
            {
                if (debug_viz)
                {
                    // annotate_frame() draws inplace
                    viz = cfg.rrd_on ? resized.clone() : resized;
                    vd::visualize(viz, *r, source_label(cfg.source), cfg.wheel_dir, pipeline.H_world2resized());
                    display_frame = viz;
                }
                else
                {
                    display_frame = visualization.build_frame(resized, *r, plan, ego_v, pipeline.H_resized(), cfg.speed_limit);
                    viz = display_frame;
                }
            }

            // Submit all required logging params to single logger func
            if (cfg.rrd_on)
                logging::Rerun::log_frame(r->frame_id, frame, warped, resized, *r, plan, ego_v, viz);
        }
        if (cfg.visualization_on)
        {
            visualization.render_frame(display_frame);
        }
    }

    if (cfg.rrd_on) logging::Rerun::shutdown();  // flush & close .rrd

    // stop() returns true on a clean shutdown; translate that to a 0 exit code
    // so VisionPilot can be supervised as a batch/oneshot job (a successful run
    // must not exit non-zero).
    return visualization.stop() ? 0 : 1;
}
