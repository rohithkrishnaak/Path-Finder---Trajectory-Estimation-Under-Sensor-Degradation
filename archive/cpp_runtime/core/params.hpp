// Tunable constants. MUST mirror research/ref_pipeline.py::Params (parity test).
#pragma once

namespace pf {

constexpr double G = 9.81;

struct Params {
    // validity
    double w_max = 35.0;
    double a_max = 160.0;
    double gap_factor = 2.5;
    // spike repair
    double spike_k = 10.0;
    double spike_floor_g = 0.3;
    double spike_floor_a = 3.0;
    // noise
    double sig_g = 2.5e-3;
    double sig_a = 3.0e-2;
    double sig_bg_rw = 2.0e-4;
    double sig_ba_rw = 3.0e-2;
    double noise_ema_tau = 1.0;
    // initialisation
    double init_buf_s = 10.0;
    int static_win = 40;
    int init_win = 100;
    double p0_att = 0.02;
    double p0_yaw = 1e-4;
    double p0_v = 0.05;
    double p0_ba = 0.1;
    double p0_bg = 0.01;
    // static detector + updates
    double static_acc_std = 0.10;
    double static_gyr_mean = 0.04;
    int static_every = 10;
    double sig_zupt = 0.02;
    double sig_tilt = 0.05;
    double sig_zrate = 0.002;
    // gap / blind propagation
    double tau_v_gap = 0.5;
    double tau_w_gap = 0.5;
    double sig_v_gap = 1.0;
    double sig_th_gap = 0.2;
    double tau_v_blind = 0.5;
    double tau_w_blind = 1.0;
    double w_hist_s = 0.5;
    double blind_s = 5.0;
    // weak velocity prior
    double sig_vprior = 0.0;   // off when the learned velocity is on (see devlog D-025)
    int vprior_every = 10;
    // learned velocity measurement
    double nn_sig_scale = 2.0;   // multiplies the predicted std
    double nn_sig_floor = 0.10;  // m/s
    int nn_every_blocks = 10;    // update every 10 blocks (40 samples = 0.2 s)
    int nn_full = 1;             // 1: update may correct all states; 0: velocity only
    double nn_noise_pow = 0.0;   // sigma inflation exponent on the estimated accel-noise ratio
    double nn_vmax = 6.0;        // reject predictions beyond this speed
};

}  // namespace pf
