// Two-pass streaming pipeline. MUST mirror research/ref_pipeline.py (scan + run).
// RAM is O(1) in sequence length: pass 1 keeps only a <=10 s init buffer, pass 2 streams with
// one row of lookahead and writes poses immediately.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "eskf.hpp"
#include "ingest.hpp"
#include "nn.hpp"
#include "params.hpp"

namespace pf {

enum class BlindMode { Auto, Append, None };

inline bool row_valid(const Row& r, const Row* prev, const Params& P) {
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(r.g[k]) || !std::isfinite(r.a[k])) return false;
    bool zero = true;
    for (int k = 0; k < 3; ++k) zero = zero && r.g[k] == 0.0 && r.a[k] == 0.0;
    if (zero) return false;
    for (int k = 0; k < 3; ++k)
        if (std::fabs(r.g[k]) > P.w_max || std::fabs(r.a[k]) > P.a_max) return false;
    if (prev) {
        bool same = true;
        for (int k = 0; k < 3; ++k) same = same && r.g[k] == prev->g[k] && r.a[k] == prev->a[k];
        if (same) return false;
    }
    return true;
}

struct ScanResult {
    int64_t n = 0;
    int64_t last_valid = -1;
    int64_t dt_ns = 5000000;
    Vec4 q0 = Vec4(1, 0, 0, 0);
    Vec3 bg0 = Vec3::Zero();
    bool static_init = false;
};

inline bool scan(const std::string& path, const Params& P, ScanResult& s, std::string& err) {
    RowReader rd;
    if (!rd.open(path, err)) return false;
    Row r, prev;
    bool has_prev = false;
    int64_t t_first = 0, t_lim = 0;
    std::vector<int64_t> diffs;
    diffs.reserve(2000);
    std::vector<Vec3> gb, ab;  // init buffer (valid rows in the first init_buf_s seconds)
    const size_t buf_cap = 8000;
    while (rd.next(r)) {
        const int64_t i = s.n;
        if (i == 0) { t_first = r.t; t_lim = t_first + (int64_t)(P.init_buf_s * 1e9); }
        else if (i <= 2000) { const int64_t d = r.t - prev.t; if (d > 0) diffs.push_back(d); }
        const bool v = row_valid(r, has_prev ? &prev : nullptr, P);
        if (v) {
            s.last_valid = i;
            if (r.t <= t_lim && gb.size() < buf_cap) {
                gb.emplace_back(r.g[0], r.g[1], r.g[2]);
                ab.emplace_back(r.a[0], r.a[1], r.a[2]);
            }
        }
        prev = r;
        has_prev = true;
        ++s.n;
    }
    if (!diffs.empty()) {
        std::sort(diffs.begin(), diffs.end());
        const size_t m = diffs.size();
        s.dt_ns = (m % 2) ? diffs[m / 2] : (int64_t)((double(diffs[m / 2 - 1]) + double(diffs[m / 2])) / 2.0);
    }
    const int W = P.init_win;
    const int nb = (int)gb.size();
    bool found = false;
    for (int k = 0; k + W <= nb; k += W / 4) {
        Vec3 gm = Vec3::Zero(), am = Vec3::Zero();
        for (int j = k; j < k + W; ++j) { gm += gb[j]; am += ab[j]; }
        gm /= W; am /= W;
        double an_mean = 0, gn_mean = 0;
        for (int j = k; j < k + W; ++j) { an_mean += ab[j].norm(); gn_mean += (gb[j] - gm).norm(); }
        an_mean /= W; gn_mean /= W;
        double an_var = 0;
        for (int j = k; j < k + W; ++j) { const double d = ab[j].norm() - an_mean; an_var += d * d; }
        const double an_std = std::sqrt(an_var / W);
        if (an_std < P.static_acc_std && gn_mean < P.static_gyr_mean && std::fabs(an_mean - G) < 0.5) {
            s.q0 = gravity_to_quat(am);
            s.bg0 = gm;
            s.static_init = true;
            found = true;
            break;
        }
    }
    if (!found && nb > 0) {
        Vec3 am = Vec3::Zero();
        for (int j = 0; j < nb; ++j) am += ab[j];
        s.q0 = gravity_to_quat(am / nb);
    }
    return true;
}

class Writer {
public:
    bool open(const std::string& path) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) return false;
        std::setvbuf(f_, nullptr, _IOFBF, 1 << 16);
        std::fputs("# timestamp tx ty tz qx qy qz qw\n", f_);
        return true;
    }
    void emit(int64_t t_ns, const Vec3& p, Vec4 q) {
        q /= q.norm();
        if (has_last_ && q.dot(last_q_) < 0) q = -q;
        last_q_ = q;
        has_last_ = true;
        int64_t sec = t_ns / 1000000000LL, frac = t_ns % 1000000000LL;
        if (frac < 0) { frac += 1000000000LL; sec -= 1; }
        std::fprintf(f_, "%lld.%09lld %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", (long long)sec, (long long)frac,
                     p[0], p[1], p[2], q[1], q[2], q[3], q[0]);
        ++count_;
    }
    bool close() { bool ok = f_ && std::fclose(f_) == 0; f_ = nullptr; return ok; }
    int64_t count() const { return count_; }
private:
    FILE* f_ = nullptr;
    Vec4 last_q_ = Vec4(1, 0, 0, 0);
    bool has_last_ = false;
    int64_t count_ = 0;
};

struct RunStats {
    int64_t rows = 0, valid_rows = 0, poses = 0, appended = 0;
    double imu_seconds = 0;
    bool static_init = false;
};

template <typename T>
inline Vec3 mean_of(const std::vector<T>& v) {
    Vec3 m = Vec3::Zero();
    if (v.empty()) return m;
    for (const auto& x : v) m += x;
    return m / double(v.size());
}

inline bool run(const std::string& in_path, const std::string& out_path, const Params& P, BlindMode mode,
                RunStats& st, std::string& err, const VelNet* net = nullptr) {
    ScanResult sc;
    if (!scan(in_path, P, sc, err)) return false;
    st.rows = sc.n;
    st.static_init = sc.static_init;
    Writer wr;
    if (!wr.open(out_path)) { err = "cannot write " + out_path; return false; }
    RowReader rd;
    if (!rd.open(in_path, err)) return false;

    ESKF f(sc.q0, sc.bg0, P, sc.static_init);
    const double dt_nom = sc.dt_ns * 1e-9;
    const double gap_dt = P.gap_factor * dt_nom;
    const double alpha = 1.0 - std::exp(-dt_nom / P.noise_ema_tau);
    double var_g = P.sig_g * P.sig_g, var_a = P.sig_a * P.sig_a;

    bool has_last = false;
    Vec3 last_g, last_a;
    std::vector<Vec3> w_hist;
    const size_t w_hist_n = (size_t)std::max<long long>(1, std::llround(P.w_hist_s / dt_nom));
    w_hist.reserve(w_hist_n + 1);
    Vec3 w_gap = Vec3::Zero();
    bool in_gap = false;
    std::vector<Vec3> win_g, win_a;
    win_g.reserve(P.static_win + 1);
    win_a.reserve(P.static_win + 1);
    long long static_count = 0;
    // learned-velocity window: 50 blocks of 4 samples, features [a*SA, g*SG, gravity-in-body]
    std::vector<float> nn_blocks((size_t)NN_T * NN_F, 0.f);
    int nn_nb = 0, nn_ns = 0, nn_since = 0;
    double nn_acc[NN_F] = {0};
    Vec3 good_p = f.p, good_v = f.v;
    Vec4 good_q = f.q;
    const int64_t n = sc.n;
    const int64_t blind_start = sc.last_valid + 1;

    // lookahead: cur = row i (with validity computed on the raw previous row), nxt = row i+1
    Row cur, nxt, raw_prev;
    bool cur_valid = false, nxt_valid = false, have_nxt = false;
    int64_t t_prev = 0, t_last_row = 0, t_first_row = 0;
    if (n > 0) {
        rd.next(cur);
        t_first_row = cur.t;
        cur_valid = row_valid(cur, nullptr, P);
        raw_prev = cur;
        if (n > 1) { rd.next(nxt); have_nxt = true; nxt_valid = row_valid(nxt, &raw_prev, P); }
    }
    for (int64_t i = 0; i < n; ++i) {
        const bool valid = cur_valid;
        if (valid) ++st.valid_rows;
        // spike repair (needs previous valid sample and a valid next sample)
        if (valid && has_last && have_nxt && nxt_valid) {
            for (int which = 0; which < 2; ++which) {
                double* x = which == 0 ? cur.g : cur.a;
                const double* xn = which == 0 ? nxt.g : nxt.a;
                const Vec3& xp = which == 0 ? last_g : last_a;
                const double T = std::max(P.spike_k * std::sqrt(which == 0 ? var_g : var_a),
                                          which == 0 ? P.spike_floor_g : P.spike_floor_a);
                for (int c = 0; c < 3; ++c) {
                    if (std::fabs(x[c] - xp[c]) > T && std::fabs(x[c] - xn[c]) > T && std::fabs(xn[c] - xp[c]) < 0.5 * T)
                        x[c] = 0.5 * (xp[c] + xn[c]);
                }
            }
        }
        const Vec3 g1(cur.g[0], cur.g[1], cur.g[2]), a1(cur.a[0], cur.a[1], cur.a[2]);
        if (i == 0) {
            if (valid) { last_g = g1; last_a = a1; has_last = true; }
            wr.emit(cur.t, f.p, f.q);
        } else {
            const double dt = (cur.t - t_prev) * 1e-9;
            if (i >= blind_start) {
                if (i == blind_start) w_gap = mean_of(w_hist);
                if (dt > 0) {
                    w_gap = w_gap * std::exp(-dt / P.tau_w_blind);
                    f.propagate_gap(w_gap, dt, P.tau_v_blind, P.sig_v_gap, P.sig_th_gap);
                }
                wr.emit(cur.t, f.p, f.q);
            } else if (dt <= 0) {
                wr.emit(cur.t, f.p, f.q);
            } else {
                if (valid && has_last && dt <= gap_dt && !in_gap) {
                    var_g = std::max(P.sig_g * P.sig_g, (1 - alpha) * var_g + alpha * 0.5 * (g1 - last_g).squaredNorm() / 3.0);
                    var_a = std::max(P.sig_a * P.sig_a, (1 - alpha) * var_a + alpha * 0.5 * (a1 - last_a).squaredNorm() / 3.0);
                    f.propagate_imu(last_g, last_a, g1, a1, dt, std::sqrt(var_g), std::sqrt(var_a));
                } else {
                    if (!in_gap) { w_gap = mean_of(w_hist); in_gap = true; }
                    w_gap = w_gap * std::exp(-dt / P.tau_w_gap);
                    f.propagate_gap(w_gap, dt, P.tau_v_gap, P.sig_v_gap, P.sig_th_gap);
                    if (valid) { in_gap = false; win_g.clear(); win_a.clear(); static_count = 0; }
                    nn_nb = 0; nn_ns = 0; nn_since = 0; for (double& x : nn_acc) x = 0;
                }
                if (valid) {
                    last_g = g1; last_a = a1; has_last = true;
                    w_hist.push_back(g1 - f.bg);
                    if (w_hist.size() > w_hist_n) w_hist.erase(w_hist.begin());
                    if (P.sig_vprior > 0 && i % P.vprior_every == 0) f.update_velocity_zero(P.sig_vprior, true);
                    if (net && net->ok && !in_gap) {
                        const Mat3 Rn = quat_to_rot(f.q);
                        const Vec3 gb = Rn.row(2).transpose();
                        const double feat[NN_F] = {a1[0] * NN_SA, a1[1] * NN_SA, a1[2] * NN_SA, g1[0] * NN_SG, g1[1] * NN_SG,
                                                   g1[2] * NN_SG, gb[0], gb[1], gb[2]};
                        for (int k = 0; k < NN_F; ++k) nn_acc[k] += feat[k];
                        if (++nn_ns == 4) {
                            nn_ns = 0;
                            if (nn_nb == NN_T) std::copy(nn_blocks.begin() + NN_F, nn_blocks.end(), nn_blocks.begin());
                            else ++nn_nb;
                            for (int k = 0; k < NN_F; ++k) { nn_blocks[(size_t)(nn_nb - 1) * NN_F + k] = float(nn_acc[k] / 4.0); nn_acc[k] = 0; }
                            if (nn_nb == NN_T && ++nn_since >= P.nn_every_blocks) {
                                nn_since = 0;
                                float out[NN_OUT];
                                net->forward(nn_blocks.data(), out);
                                const Vec3 vb(out[0], out[1], out[2]);
                                if (vb.norm() < P.nn_vmax) {
                                    const double var = (std::exp(out[3]) + std::exp(out[4]) + std::exp(out[5])) / 3.0;
                                    const double infl = std::pow(std::max(1.0, std::sqrt(var_a) / P.sig_a), P.nn_noise_pow);
                                    const double sig = std::max(P.nn_sig_floor, P.nn_sig_scale * std::sqrt(var)) * infl;
                                    const Vec3 vw = Rn * vb;
                                    Mat3x15 H = Mat3x15::Zero();
                                    H.block<3, 3>(0, IV) = Mat3::Identity();
                                    if (P.nn_full) f.update(H, vw - f.v, sig);
                                    else f.update(H, vw - f.v, sig, IV, IV + 3);
                                }
                            }
                        }
                    }
                    win_g.push_back(g1); win_a.push_back(a1);
                    if ((int)win_g.size() > P.static_win) { win_g.erase(win_g.begin()); win_a.erase(win_a.begin()); }
                    if ((int)win_g.size() == P.static_win) {
                        const int W = P.static_win;
                        double an_mean = 0, gn_mean = 0;
                        for (int j = 0; j < W; ++j) { an_mean += win_a[j].norm(); gn_mean += (win_g[j] - f.bg).norm(); }
                        an_mean /= W; gn_mean /= W;
                        double an_var = 0;
                        for (int j = 0; j < W; ++j) { const double d = win_a[j].norm() - an_mean; an_var += d * d; }
                        if (std::sqrt(an_var / W) < P.static_acc_std && gn_mean < P.static_gyr_mean) {
                            ++static_count;
                            if (static_count % P.static_every == 0) f.static_updates(mean_of(win_a), mean_of(win_g));
                        } else {
                            static_count = 0;
                        }
                    }
                }
                if (!f.finite()) {
                    f.p = good_p; f.v = good_v; f.q = good_q;
                    f.P = Mat15::Identity();
                } else {
                    good_p = f.p; good_v = f.v; good_q = f.q;
                }
                wr.emit(cur.t, f.p, f.q);
            }
        }
        // advance
        t_prev = cur.t;
        t_last_row = cur.t;
        if (have_nxt) {
            raw_prev = nxt;  // raw copy (spike repair only ever touches `cur`)
            cur = nxt;
            cur_valid = nxt_valid;
            have_nxt = (i + 2 < n) && rd.next(nxt);
            if (have_nxt) nxt_valid = row_valid(nxt, &raw_prev, P);
        }
    }
    if (n > 0 && (mode == BlindMode::Append || (mode == BlindMode::Auto && blind_start >= n))) {
        w_gap = mean_of(w_hist);
        const double dt = sc.dt_ns * 1e-9;
        const int64_t lim = (int64_t)(P.blind_s * 1e9);
        for (int64_t k = 1; k * sc.dt_ns <= lim; ++k) {
            w_gap = w_gap * std::exp(-dt / P.tau_w_blind);
            f.propagate_gap(w_gap, dt, P.tau_v_blind, P.sig_v_gap, P.sig_th_gap);
            wr.emit(t_last_row + k * sc.dt_ns, f.p, f.q);
            ++st.appended;
        }
    }
    st.poses = wr.count();
    st.imu_seconds = n > 1 ? (t_last_row - t_first_row) * 1e-9 : 0.0;
    if (!wr.close()) { err = "write failed"; return false; }
    return true;
}

}  // namespace pf
