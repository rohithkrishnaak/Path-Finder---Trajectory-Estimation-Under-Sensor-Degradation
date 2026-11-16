// 15-state error-state Kalman filter. MUST mirror research/ref_pipeline.py::ESKF.
// Error state order: [dp(3) dv(3) dtheta(3) dba(3) dbg(3)], local (right) angle error.
#pragma once

#include <cmath>

#include <Eigen/Dense>

#include "params.hpp"

namespace pf {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec4 = Eigen::Vector4d;  // quaternion stored [w x y z]
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Mat3x15 = Eigen::Matrix<double, 3, 15>;
using Mat15x3 = Eigen::Matrix<double, 15, 3>;

enum { IP = 0, IV = 3, ITH = 6, IBA = 9, IBG = 12 };

inline Mat3 skew(const Vec3& v) {
    Mat3 m;
    m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return m;
}

inline Vec4 quat_mul(const Vec4& a, const Vec4& b) {
    const double aw = a[0], ax = a[1], ay = a[2], az = a[3];
    const double bw = b[0], bx = b[1], by = b[2], bz = b[3];
    return Vec4(aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw);
}

inline Vec4 quat_exp(const Vec3& phi) {
    const double ang = std::sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);
    if (ang < 1e-12) {
        Vec4 q(1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]);
        return q / q.norm();
    }
    const double s = std::sin(0.5 * ang) / ang;
    return Vec4(std::cos(0.5 * ang), phi[0] * s, phi[1] * s, phi[2] * s);
}

inline Mat3 quat_to_rot(const Vec4& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    Mat3 R;
    R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
    return R;
}

// Roll/pitch from specific force (body frame), yaw = 0. q = Ry(pitch) * Rx(roll).
inline Vec4 gravity_to_quat(const Vec3& f) {
    const double roll = std::atan2(f[1], f[2]);
    const double pitch = std::atan2(-f[0], std::sqrt(f[1] * f[1] + f[2] * f[2]));
    const double cr = std::cos(roll / 2), sr = std::sin(roll / 2), cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
    return Vec4(cp * cr, cp * sr, sp * cr, -sp * sr);
}

struct ESKF {
    const Params& prm;
    Vec3 p = Vec3::Zero(), v = Vec3::Zero(), ba = Vec3::Zero(), bg = Vec3::Zero();
    Vec4 q = Vec4(1, 0, 0, 0);
    Mat15 P = Mat15::Zero();
    const Vec3 gvec = Vec3(0.0, 0.0, -G);

    ESKF(const Vec4& q0, const Vec3& bg0, const Params& P_, bool static_init) : prm(P_) {
        q = q0;
        bg = bg0;
        Vec15 d = Vec15::Zero();
        for (int i = 0; i < 3; ++i) d[IP + i] = 1e-6;
        const double sv = static_init ? prm.p0_v : 1.0;
        for (int i = 0; i < 3; ++i) d[IV + i] = sv * sv;
        d[ITH] = d[ITH + 1] = prm.p0_att * prm.p0_att;
        d[ITH + 2] = prm.p0_yaw * prm.p0_yaw;
        for (int i = 0; i < 3; ++i) { d[IBA + i] = prm.p0_ba * prm.p0_ba; d[IBG + i] = prm.p0_bg * prm.p0_bg; }
        P = d.asDiagonal();
    }

    void add_diag(const Vec15& qd) { for (int i = 0; i < 15; ++i) P(i, i) += qd[i]; }

    void propagate_imu(const Vec3& g0, const Vec3& a0, const Vec3& g1, const Vec3& a1, double dt, double sig_g, double sig_a) {
        const Mat3 R0 = quat_to_rot(q);
        const Vec3 w = 0.5 * (g0 + g1) - bg;
        Vec4 q1 = quat_mul(q, quat_exp(w * dt));
        q1 /= q1.norm();
        const Mat3 R1 = quat_to_rot(q1);
        const Vec3 aW = 0.5 * (R0 * (a0 - ba) + R1 * (a1 - ba)) + gvec;
        p = p + v * dt + 0.5 * aW * dt * dt;
        v = v + aW * dt;
        q = q1;
        const Vec3 ab = 0.5 * (a0 + a1) - ba;
        Mat15 F = Mat15::Identity();
        const Mat3 RA = R0 * skew(ab);
        F.block<3, 3>(IP, IV) = Mat3::Identity() * dt;
        F.block<3, 3>(IP, ITH) = -0.5 * RA * dt * dt;
        F.block<3, 3>(IP, IBA) = -0.5 * R0 * dt * dt;
        F.block<3, 3>(IV, ITH) = -RA * dt;
        F.block<3, 3>(IV, IBA) = -R0 * dt;
        F.block<3, 3>(ITH, ITH) = quat_to_rot(quat_exp(w * dt)).transpose();
        F.block<3, 3>(ITH, IBG) = -Mat3::Identity() * dt;
        Vec15 qd = Vec15::Zero();
        for (int i = 0; i < 3; ++i) {
            qd[IV + i] = (sig_a * dt) * (sig_a * dt);
            qd[ITH + i] = (sig_g * dt) * (sig_g * dt);
            qd[IBA + i] = prm.sig_ba_rw * prm.sig_ba_rw * dt;
            qd[IBG + i] = prm.sig_bg_rw * prm.sig_bg_rw * dt;
        }
        P = F * P * F.transpose();
        add_diag(qd);
    }

    void propagate_gap(const Vec3& w_body, double dt, double tau_v, double sig_v, double sig_th) {
        const double dv = std::exp(-dt / tau_v);
        p = p + v * dt;
        v = v * dv;
        q = quat_mul(q, quat_exp(w_body * dt));
        q /= q.norm();
        Mat15 F = Mat15::Identity();
        F.block<3, 3>(IP, IV) = Mat3::Identity() * dt;
        F.block<3, 3>(IV, IV) = Mat3::Identity() * dv;
        Vec15 qd = Vec15::Zero();
        for (int i = 0; i < 3; ++i) {
            qd[IV + i] = sig_v * sig_v * dt;
            qd[ITH + i] = sig_th * sig_th * dt;
            qd[IBA + i] = prm.sig_ba_rw * prm.sig_ba_rw * dt;
            qd[IBG + i] = prm.sig_bg_rw * prm.sig_bg_rw * dt;
        }
        P = F * P * F.transpose();
        add_diag(qd);
    }

    // EKF update; [only_lo, only_hi) restricts the correction to those error states (Schmidt-style).
    void update(const Mat3x15& H, const Vec3& r, double sig, int only_lo = 0, int only_hi = 15) {
        const Mat3 Rm = Mat3::Identity() * (sig * sig);
        const Mat3 S = H * P * H.transpose() + Rm;
        Mat15x3 K = P * H.transpose() * S.inverse();
        for (int i = 0; i < 15; ++i)
            if (i < only_lo || i >= only_hi) K.row(i).setZero();
        const Vec15 dx = K * r;
        const Mat15 IKH = Mat15::Identity() - K * H;
        P = IKH * P * IKH.transpose() + K * Rm * K.transpose();
        P = 0.5 * (P + P.transpose()).eval();
        p += dx.segment<3>(IP);
        v += dx.segment<3>(IV);
        q = quat_mul(q, quat_exp(dx.segment<3>(ITH)));
        q /= q.norm();
        ba += dx.segment<3>(IBA);
        bg += dx.segment<3>(IBG);
    }

    void update_velocity_zero(double sig, bool velocity_only = false) {
        Mat3x15 H = Mat3x15::Zero();
        H.block<3, 3>(0, IV) = Mat3::Identity();
        if (velocity_only) update(H, -v, sig, IV, IV + 3);
        else update(H, -v, sig);
    }

    void static_updates(const Vec3& a_mean, const Vec3& g_mean) {
        update_velocity_zero(prm.sig_zupt);
        const Mat3 R = quat_to_rot(q);
        const Vec3 fb = R.transpose() * Vec3(0.0, 0.0, G);
        Mat3x15 H = Mat3x15::Zero();
        H.block<3, 3>(0, ITH) = skew(fb);
        H.block<3, 3>(0, IBA) = Mat3::Identity();
        update(H, a_mean - (ba + fb), prm.sig_tilt);
        H.setZero();
        H.block<3, 3>(0, IBG) = Mat3::Identity();
        update(H, g_mean - bg, prm.sig_zrate);
    }

    bool finite() const { return p.allFinite() && q.allFinite() && P.allFinite(); }
};

}  // namespace pf
