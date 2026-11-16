// Learned body-frame velocity model: 1 s of IMU (50 blocks x 9 features) -> velocity (3) + log-variance (3).
// Weights are loaded from a flat float32 file (PF_WEIGHTS / --weights) or the embedded weights.hpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace pf {

constexpr int NN_T = 50, NN_F = 9, NN_IN = NN_T * NN_F, NN_H1 = 256, NN_H2 = 128, NN_OUT = 6;
constexpr double NN_SA = 0.2, NN_SG = 0.5;

// Weights are kept as bfloat16 (half the RAM); they are widened to float on the fly.
using bf16 = uint16_t;
inline bf16 to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
    return (bf16)(u >> 16);
}
inline float from_bf16(bf16 h) {
    const uint32_t u = (uint32_t)h << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

struct MLP {
    std::vector<bf16> w1, w3, w2;
    std::vector<float> b1, b2, b3;  // biases stay float (tiny)

    size_t n_floats() const { return (size_t)NN_H1 * NN_IN + NN_H1 + (size_t)NN_H2 * NN_H1 + NN_H2 + (size_t)NN_OUT * NN_H2 + NN_OUT; }

    void forward(const float* x, float* out) const {
        float h1[NN_H1], h2[NN_H2];
        for (int o = 0; o < NN_H1; ++o) {
            float s = b1[o];
            const bf16* w = &w1[(size_t)o * NN_IN];
            for (int i = 0; i < NN_IN; ++i) s += from_bf16(w[i]) * x[i];
            h1[o] = s > 0 ? s : 0;
        }
        for (int o = 0; o < NN_H2; ++o) {
            float s = b2[o];
            const bf16* w = &w2[(size_t)o * NN_H1];
            for (int i = 0; i < NN_H1; ++i) s += from_bf16(w[i]) * h1[i];
            h2[o] = s > 0 ? s : 0;
        }
        for (int o = 0; o < NN_OUT; ++o) {
            float s = b3[o];
            const bf16* w = &w3[(size_t)o * NN_H2];
            for (int i = 0; i < NN_H2; ++i) s += from_bf16(w[i]) * h2[i];
            out[o] = s;
        }
    }
};

// Ensemble of K identical-shape MLPs stored back to back. Output: mean velocity (3) and
// log of (mean predicted variance + variance between members) (3).
struct VelNet {
    std::vector<MLP> m;
    bool ok = false;

    // Streams the file member by member (no second copy in RAM).
    bool load_file(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        MLP probe;
        const size_t n = probe.n_floats();
        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (bytes <= 0 || (size_t)bytes % (4 * n) != 0) { std::fclose(f); return false; }
        m.assign((size_t)bytes / (4 * n), MLP());
        bool good = true;
        std::vector<float> tmp;
        auto rd = [&](auto& dst, size_t len) {
            tmp.resize(len);
            good = good && std::fread(tmp.data(), 4, len, f) == len;
            dst.resize(len);
            for (size_t i = 0; i < len; ++i) {
                if constexpr (std::is_same<typename std::remove_reference<decltype(dst)>::type, std::vector<bf16>>::value) dst[i] = to_bf16(tmp[i]);
                else dst[i] = tmp[i];
            }
        };
        for (auto& net : m) {
            rd(net.w1, (size_t)NN_H1 * NN_IN); rd(net.b1, NN_H1); rd(net.w2, (size_t)NN_H2 * NN_H1); rd(net.b2, NN_H2);
            rd(net.w3, (size_t)NN_OUT * NN_H2); rd(net.b3, NN_OUT);
        }
        std::fclose(f);
        return ok = good;
    }

    void forward(const float* x, float* out) const {
        const int K = (int)m.size();
        float o[NN_OUT], mean[3] = {0, 0, 0}, var[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
        for (const auto& net : m) {
            net.forward(x, o);
            for (int c = 0; c < 3; ++c) { mean[c] += o[c]; sq[c] += o[c] * o[c]; var[c] += std::exp(o[3 + c]); }
        }
        for (int c = 0; c < 3; ++c) {
            mean[c] /= K; var[c] /= K;
            const float between = std::max(0.0f, sq[c] / K - mean[c] * mean[c]);
            out[c] = mean[c];
            out[3 + c] = std::log(var[c] + between);
        }
    }
};

}  // namespace pf
