// Non-interactive contract test for LongitudinalPlanner.
//
// CONTRACT UNDER TEST: cipo_v is the ABSOLUTE lead-vehicle speed in m/s
// (speed_limit when free road). The planner derives the IDM approach rate
// internally as delta_v = ego_v - cipo_v (positive when closing).
//
// Regression: the runtime used to forward the fusion RELATIVE velocity
// (negative when approaching) straight into cipo_v while one line read it
// as relative and every other consumer (headers, horizon, sim) read it as
// absolute. With relative input the opening case below brakes hard while
// the lead pulls away (computed -5.3 m/s^2 pre-fix vs +1.5 post-fix).
//
// Build (no ROS/ORT needed, planner is dependency-free):
//   g++ -I modules/safety_guardian/planning/include \
//       modules/safety_guardian/planning/src/longitudinal_planning.cpp \
//       tests/safety_guardian/planning/test_longitudinal_planning.cpp \
//       -o /tmp/test_longitudinal && /tmp/test_longitudinal
#include <cmath>
#include <cstdio>
#include <planning/longitudinal_planning.hpp>

static int failures = 0;

static void check(bool ok, const char* name, double accel) {
    std::printf("%-42s accel=%+7.3f  %s\n", name, accel, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

int main() {
    LongitudinalPlanner::Config cfg;
    cfg.speed_limit = 30.0;
    LongitudinalPlanner planner(cfg);
    const double kappa = 0.0;  // straight road

    // 1. Opening: lead pulls away (ego 10, lead 20, gap 30).
    //    Must cruise near free-road accel (~+1.5). Pre-fix: -5.3 (slams brakes).
    {
        double a = planner.compute_acceleration(kappa, 10.0, true, 20.0, 30.0);
        check(a > 0.5 && a < 2.0, "opening gap cruises", a);
    }

    // 2. Closing: ego faster than lead (ego 20, lead 12, gap 30).
    //    Must brake firmly but sanely (~-6.9). Pre-fix: -11.9 (over-brake).
    {
        double a = planner.compute_acceleration(kappa, 20.0, true, 12.0, 30.0);
        check(a < -2.0 && a > -10.0, "closing gap brakes firmly", a);
    }

    // 3. Free road, no lead: full comfort accel (~+1.5).
    {
        double a = planner.compute_acceleration(kappa, 5.0, false, 30.0, 9999.0);
        check(a > 1.0 && a < 1.6, "free road accelerates", a);
    }

    // 4. Stopped lead, close gap (ego 10, lead 0, gap 8): must brake hard.
    {
        double a = planner.compute_acceleration(kappa, 10.0, true, 0.0, 8.0);
        check(a < -5.0, "stopped lead brakes hard", a);
    }

    // 5. Standstill, free road: full accel authority.
    {
        double a = planner.compute_acceleration(kappa, 0.0, false, 30.0, 9999.0);
        check(a > 1.4 && a < 1.6, "standstill launches", a);
    }

    // 6. Degenerate inputs stay finite (gap floor, curved road).
    {
        double a1 = planner.compute_acceleration(kappa, 10.0, true, 9.0, 0.5);
        double a2 = planner.compute_acceleration(0.3, 10.0, false, 30.0, 9999.0);
        check(std::isfinite(a1) && std::isfinite(a2), "degenerate finite", a1);
    }

    if (failures == 0) {
        std::printf("ALL LONGITUDINAL CONTRACT TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", failures);
    return 1;
}
