// Path Finder - IMU-only 6-DoF trajectory estimation under sensor degradation (Samsung PRISM XR Hackathon 2026, Track 3).
//
//   pathfinder <input.zip | folder | imu0/data.csv> [-o estimated_trajectory.txt] [--blind auto|append|none]
//              [--metrics metrics.json]
//
// Single thread, streaming, O(1) RAM in sequence length. See README.md and docs/plan.md.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "core/pipeline.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

static double peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return -1;
#else
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    double kb = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) { kb = std::atof(line + 6); break; }
    }
    std::fclose(f);
    return kb / 1024.0;
#endif
}

static void usage() {
    std::fprintf(stderr,
                 "usage: pathfinder <input.zip|folder|csv> [-o estimated_trajectory.txt] [--blind auto|append|none] "
                 "[--metrics metrics.json]\n");
}

int main(int argc, char** argv) {
    std::string in, out = "estimated_trajectory.txt", metrics;
    pf::BlindMode mode = pf::BlindMode::Auto;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-o" || a == "--output") && i + 1 < argc) out = argv[++i];
        else if (a == "--blind" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "auto") mode = pf::BlindMode::Auto;
            else if (m == "append") mode = pf::BlindMode::Append;
            else if (m == "none") mode = pf::BlindMode::None;
            else { usage(); return 2; }
        } else if (a == "--metrics" && i + 1 < argc) metrics = argv[++i];
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (in.empty()) in = a;
        else { usage(); return 2; }
    }
    if (in.empty()) { usage(); return 2; }

    const auto t0 = std::chrono::steady_clock::now();
    pf::Params P;
    pf::RunStats st;
    std::string err;
    if (!pf::run(in, out, P, mode, st, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double ms_per_s = st.imu_seconds > 0 ? ms / st.imu_seconds : 0.0;
    const double rss = peak_rss_mb();
    std::fprintf(stderr,
                 "pathfinder: rows=%lld valid=%lld poses=%lld appended=%lld imu=%.1fs static_init=%d | "
                 "%.1f ms total, %.3f ms per IMU-second, peak RSS %.2f MB\n",
                 (long long)st.rows, (long long)st.valid_rows, (long long)st.poses, (long long)st.appended,
                 st.imu_seconds, (int)st.static_init, ms, ms_per_s, rss);
    if (!metrics.empty()) {
        if (FILE* f = std::fopen(metrics.c_str(), "w")) {
            std::fprintf(f,
                         "{\n \"rows\": %lld,\n \"valid_rows\": %lld,\n \"poses\": %lld,\n \"appended\": %lld,\n"
                         " \"imu_seconds\": %.3f,\n \"wall_ms\": %.3f,\n \"ms_per_imu_second\": %.4f,\n"
                         " \"peak_rss_mb\": %.3f,\n \"threads\": 1\n}\n",
                         (long long)st.rows, (long long)st.valid_rows, (long long)st.poses, (long long)st.appended,
                         st.imu_seconds, ms, ms_per_s, rss);
            std::fclose(f);
        }
    }
    return 0;
}
