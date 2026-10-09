#!/usr/bin/env python3
"""绘制开环 Bode 图并读出相位裕度。

对应文档 docs/phase_margin_analysis.md。

开环传递函数（PD 控制器 + 双积分器被控对象 + 纯指令延时）:

    L(s) = (kp + kd*s) / (I*s^2 + B*s) * exp(-s*T)

    I_total  下游连杆等效惯量 (frequency_damping_ratio_calculate.py)
    B        关节粘性摩擦真值 (data_collection.py 注入的 damping)
    T        指令延时 [s] = time_lag * sim.dt，data_collection.py 里 time_lag=5 @400Hz = 12.5ms

相位裕度:
    PM = 180 + angle(L(j*wc))，其中 wc 为 |L|=1 的增益穿越频率

用法:
    python scripts/bode_phase_margin.py                      # ELBOW_YAW 默认对比
    python scripts/bode_phase_margin.py --joint HIP_YAW
    python scripts/bode_phase_margin.py --out /tmp/x.png

注意: 本脚本只做频域分析，不启动 Isaac Sim。
"""

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

# ── 各关节的 I_total 与粘性摩擦真值（与 frequency_damping_ratio_calculate.py /
#    data_collection.py 保持一致）───────────────────────────────────────────
PLANT = {
    "HIP_PITCH":      (3.8906, 1.6),
    "HIP_ROLL":       (4.5621, 1.6),
    "HIP_YAW":        (0.0858, 1.0),
    "KNEE_PITCH":     (2.2483, 1.6),
    "ANKLE_PITCH":    (0.5303, 0.5),
    "ANKLE_ROLL":     (0.0415, 0.5),
    "TORSO_YAW":      (1.0572, 1.0),
    "SHOULDER_PITCH": (0.4166, 0.5),
    "SHOULDER_ROLL":  (0.7207, 0.5),
    "SHOULDER_YAW":   (0.0441, 0.5),
    "ELBOW_PITCH":    (0.4166, 0.5),
    "ELBOW_YAW":      (0.0103, 0.1),
    "WRIST_PITCH":    (0.0554, 0.1),
    "WRIST_ROLL":     (0.0148, 0.1),
}

DELAY_S = 0.0075  # time_lag = 3 步 @ 400 Hz（须与 data_collection.py 的 time_lag 一致）

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a69e"
S1, S2 = "#2a78d6", "#eb6834"  # 分类色板 slot 1 / 2


def open_loop(w, kp, kd, inertia, viscous, delay=DELAY_S):
    """L(jw) = (kp + kd*s)/(I*s^2 + B*s) * exp(-s*T)"""
    s = 1j * w
    return (kp + kd * s) / (inertia * s**2 + viscous * s) * np.exp(-s * delay)


def crossover(w, mag_db):
    """增益穿越频率（|L| = 1 即 0 dB）。"""
    k = int(np.argmin(np.abs(mag_db)))
    return w[k]


def phase_margin(kp, kd, joint):
    """解析算出 ωc / PM，用于表格打印。"""
    inertia, viscous = PLANT[joint]
    w = 2 * np.pi * np.logspace(np.log10(0.1), np.log10(100), 6000)
    mag_db = 20 * np.log10(np.abs(open_loop(w, kp, kd, inertia, viscous)))
    ph = np.unwrap(np.angle(open_loop(w, kp, kd, inertia, viscous))) * 180 / np.pi
    k = int(np.argmin(np.abs(mag_db)))
    return w[k] / 2 / np.pi, ph[k], 180 + ph[k]


def plot(joint, configs, out_path):
    inertia, viscous = PLANT[joint]
    w = 2 * np.pi * np.logspace(np.log10(0.1), np.log10(100), 6000)

    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if os.path.exists(f):
            fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 7.8), sharex=True, gridspec_kw=dict(hspace=0.12)
    )
    fig.patch.set_facecolor(SURF)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        ax.grid(True, which="major", color=MUTED, lw=0.6, alpha=0.45)
        ax.grid(True, which="minor", color=MUTED, lw=0.35, alpha=0.22)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)

    ax1.axhline(0, color=INK2, lw=1.0, ls="--", alpha=0.75)
    ax1.text(0.105, 2, "0 dB —— |L|=1 的临界线", color=INK2, fontsize=8.5, va="bottom")
    ax2.axhline(-180, color=INK2, lw=1.0, ls="--", alpha=0.75)
    ax2.text(0.105, -174, "−180° —— 负反馈变正反馈的临界线", color=INK2, fontsize=8.5, va="bottom")

    for label, kp, kd, c in configs:
        Lj = open_loop(w, kp, kd, inertia, viscous)
        mag_db = 20 * np.log10(np.abs(Lj))
        ph = np.unwrap(np.angle(Lj)) * 180 / np.pi
        ax1.semilogx(w / 2 / np.pi, mag_db, color=c, lw=2.0, solid_capstyle="round")
        ax2.semilogx(w / 2 / np.pi, ph, color=c, lw=2.0, solid_capstyle="round")
        wc = crossover(w, mag_db)
        pc = ph[int(np.argmin(np.abs(mag_db)))]
        ax1.plot([wc / 2 / np.pi], [0], "o", ms=8, color=c, mec=SURF, mew=2, zorder=5)
        ax2.plot([wc / 2 / np.pi], [pc], "o", ms=8, color=c, mec=SURF, mew=2, zorder=5)
        ax1.annotate(
            f"ωc = {wc / 2 / np.pi:.1f} Hz", (wc / 2 / np.pi, 0),
            textcoords="offset points", xytext=(9, 12 if c == S1 else -24),
            color=INK, fontsize=10, fontweight="bold",
        )
        ax2.annotate(
            f"PM = {180 + pc:+.1f}°", (wc / 2 / np.pi, pc),
            textcoords="offset points", xytext=(11, -4 if pc < -180 else -20),
            color=INK, fontsize=10, fontweight="bold",
        )

    ax1.set_ylabel("|L|   [dB]", color=INK, fontsize=11)
    ax1.set_ylim(-60, 60)
    ax2.set_ylabel("∠L   [deg]", color=INK, fontsize=11)
    ax2.set_ylim(-290, -80)
    ax2.set_xlabel("频率  [Hz]   （对数轴）", color=INK, fontsize=11)
    ax2.set_xlim(0.1, 100)

    handles = [plt.Line2D([], [], color=c, lw=2.2, label=lb) for lb, _, _, c in configs]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=10, labelcolor=INK)
    fig.text(0.5, 0.995, f"{joint} 开环 Bode 图 —— 同一被控对象，仅控制器增益不同",
             ha="center", va="top", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.002,
             f"被控对象 G(s)=1/(I·s²+B·s)，I={inertia}，B={viscous}；"
             f"指令延时 T={DELAY_S*1000:.1f} ms；PM = 180° + ∠L(ωc)",
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.subplots_adjust(top=0.885, bottom=0.095, left=0.10, right=0.97)
    fig.savefig(out_path, dpi=170, facecolor=SURF)
    print(f"saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", default="ELBOW_YAW", choices=sorted(PLANT),
                   help="被控对象（切换 I_total / B）")
    p.add_argument("--kp1", type=float, default=50.0, help="曲线 1 kp（默认 ELBOW_YAW 方案 C）")
    p.add_argument("--kd1", type=float, default=0.474, help="曲线 1 kd")
    p.add_argument("--label1", default="方案 C", help="曲线 1 图例名")
    p.add_argument("--kp2", type=float, default=3.99, help="曲线 2 kp（默认 ELBOW_YAW 修正后）")
    p.add_argument("--kd2", type=float, default=0.0621, help="曲线 2 kd")
    p.add_argument("--label2", default="修正后", help="曲线 2 图例名")
    p.add_argument("--out", default="/tmp/bode.png")
    args = p.parse_args()

    cfg = [
        (f"{args.label1}  kp={args.kp1:g}  kd={args.kd1:g}", args.kp1, args.kd1, S1),
        (f"{args.label2}  kp={args.kp2:g}  kd={args.kd2:g}", args.kp2, args.kd2, S2),
    ]

    print(f"被控对象: {args.joint}   I_total={PLANT[args.joint][0]}  B={PLANT[args.joint][1]}   T={DELAY_S*1000:.1f}ms")
    print(f"{'配置':38s} {'ωc [Hz]':>9s} {'∠L(ωc)':>9s} {'PM':>8s}")
    for label, kp, kd, _ in cfg:
        wc, ph, pm = phase_margin(kp, kd, args.joint)
        flag = "  <-- 不稳定" if pm < 0 else ("  <-- 边缘" if pm < 30 else "")
        print(f"{label:38s} {wc:9.2f} {ph:8.1f}° {pm:7.1f}°{flag}")
    plot(args.joint, cfg, args.out)


if __name__ == "__main__":
    main()
