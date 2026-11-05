#!/usr/bin/env python3
"""Plot SDK raw.csv captures without Isaac Sim, including interrupted captures.

Each recording gets plots/{position,torque,velocity}.png and diagnostic reports.
Defaults to phase 2 (including rest) and the excited joints. No resampling,
row shifting, inferred friction, or modification of the original recording.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_all_joint_torque(directory):
    """Include all recorded SDK joints and all phases, without assuming limits."""
    plan = yaml.safe_load((directory / "experiment.yaml").read_text())
    status = yaml.safe_load((directory / "status.yaml").read_text())
    with (directory / "raw.csv").open() as file:
        header = next(csv.reader(file))
        raw = np.loadtxt(file, delimiter=",", ndmin=2)
    if raw.shape[1] != len(header) or not np.isfinite(raw).all():
        raise ValueError(f"{directory}: malformed or nonfinite raw.csv")
    names = plan["full_joint_names"]
    select = lambda key: raw[:, [header.index(f"{key}/{n}") for n in names]]
    q, target, vel, torque, kp, kd = [select(k) for k in ("q", "q_des", "qd", "tau_feedback", "kp", "kd")]
    pd = kp * (target - q) - kd * vel
    time = raw[:, header.index("monotonic_s")]
    time = time - time[0]
    phase = raw[:, header.index("phase")]
    changes = np.r_[0, np.flatnonzero(np.diff(phase)) + 1, len(raw)]
    spans = [(int(phase[a]), time[a], time[b] if b < len(raw) else time[-1])
             for a, b in zip(changes[:-1], changes[1:])]
    fig, axes = plt.subplots((len(names) + 3) // 4, 4,
                             figsize=(20, 3 * ((len(names) + 3) // 4)), sharex=True, squeeze=False)
    metrics = []
    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        locked = "WRIST" in name or "TORSO" in name
        ax.plot(time, torque[:, i], label="SDK feedback", linewidth=.8)
        ax.plot(time, pd[:, i], "--", label="nominal PD (recomputed)", linewidth=.8, alpha=.85)
        for phase_id, start, end in spans:
            if phase_id in (0, 1):
                ax.axvspan(start, end, color="gray" if phase_id == 0 else "gold", alpha=.12)
            if start > 0:
                ax.axvline(start, color="gray", linestyle=":", linewidth=.6)
        if status["result"] != "complete":
            ax.axvline(time[-1], color="red", linestyle=":", linewidth=.8)
        ax.set_title(name + (" [locked in PACE]" if locked else ""), fontsize=9)
        ax.set_xlabel("Recording time [s]", fontsize=8)
        ax.set_ylabel("Torque [Nm]", fontsize=8)
        ax.set_xlim(0, time[-1] + float(plan["sample_dt"]))
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
        metrics.append(dict(joint=name, locked_in_pace=locked,
            peak_abs_feedback_Nm=float(np.max(np.abs(torque[:, i]))),
            peak_abs_nominal_pd_Nm=float(np.max(np.abs(pd[:, i])))))
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"S800 - All {len(names)} SDK joints - Feedback and nominal PD torque\n"
                 f"{directory.name}\n{status['result']}\n"
                 "Gray: entry; yellow: gain transition; unshaded: sweep/rest. "
                 "PD = kp*(q_des-q)-kd*qd; fault-triggering sample is not recorded.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .965))
    out = directory / "plots"
    out.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"torque_all_joints.{suffix}", dpi=150)
    plt.close(fig)
    with (out / "torque_all_joints.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved all {len(names)} joints: {out / 'torque_all_joints.png'}")


def analyze(directory, all_joints=False):
    plan = yaml.safe_load((directory / "experiment.yaml").read_text())
    status = yaml.safe_load((directory / "status.yaml").read_text())
    with (directory / "raw.csv").open() as file:
        header = next(csv.reader(file))
        raw = np.loadtxt(file, delimiter=",", ndmin=2)
    if raw.shape[1] != len(header) or not np.isfinite(raw).all():
        raise ValueError(f"{directory}: malformed or nonfinite raw.csv")
    col = {name: i for i, name in enumerate(header)}
    sweep = raw[raw[:, col["phase"]] == 2]
    if len(sweep) < 2:
        raise ValueError(f"{directory}: fewer than two phase-2 samples")
    names = plan["joint_names"]
    select = lambda key, joints=names: sweep[:, [col[f"{key}/{n}"] for n in joints]]
    q, target, vel, torque, kp, kd = [select(k) for k in ("q", "q_des", "qd", "tau_feedback", "kp", "kd")]
    dt = float(plan["sample_dt"])
    sample = sweep[:, col["sample"]]
    stamp = sweep[:, col["monotonic_s"]]
    # Actual runner timestamps; this is not hardware sensor acquisition time.
    time = stamp - stamp[0] + sample[0] * dt
    active = (sample * dt >= plan["rest"]) & (sample * dt < plan["rest"] + plan["duration"])
    if not active.any():
        raise ValueError(f"{directory}: no active chirp samples")
    planned = np.loadtxt(directory / "trajectory.csv", delimiter=",", skiprows=1, ndmin=2)
    valid_indices = bool(np.all(sample == sample.astype(int)) and sample.min() >= 0 and sample.max() < len(planned))
    target_match = bool(valid_indices and planned.shape[1] == len(names) + 1 and
                        np.allclose(target, planned[sample.astype(int), 1:], atol=1e-10, rtol=0))
    interval = np.diff(stamp)
    grid_error = stamp - stamp[0] - (sample - sample[0]) * dt
    complete_samples = bool(len(sweep) == plan["sample_count"] and
                            np.array_equal(sample, np.arange(plan["sample_count"])))
    gains_match = all(np.allclose(select(key), np.asarray(plan[key])[None, :], atol=1e-10, rtol=0)
                      for key in ("kp", "kd"))
    fixed = [n for n in plan["full_joint_names"] if "WRIST" in n or "TORSO" in n]
    fixed_max = {n: float(np.max(np.abs(select("q", [n])))) for n in fixed}
    pd_proxy = kp * (target - q) - kd * vel
    effort = np.asarray(plan["effort_limit"])
    pd_ratio = np.abs(pd_proxy) / effort
    peak = np.unravel_index(np.argmax(pd_ratio), pd_ratio.shape)
    last = int(np.argmax(pd_ratio[-1]))
    checks = {
        "status_complete": status["result"] == "complete",
        "row_count_matches_status": len(raw) == status["rows"],
        "finite_values": True,
        "raw_ticks_contiguous": bool(np.array_equal(raw[:, col["tick"]], np.arange(len(raw)))),
        "sweep_samples_complete": complete_samples,
        "recorded_sweep_prefix_contiguous": bool(np.array_equal(sample, np.arange(len(sweep)))),
        "timestamps_monotonic": bool(np.all(interval > 0)),
        "timing_within_1ms": bool(np.max(np.abs(interval - dt)) <= .001 and np.max(np.abs(grid_error)) <= .001),
        "logged_targets_match_plan": target_match,
        "logged_gains_match_plan": bool(gains_match),
        "fixed_joint_error_within_manifest_limit": bool(max(fixed_max.values(), default=0) <= plan["fixed_joint_error_limit"]),
    }
    metrics = []
    for i, name in enumerate(names):
        qa, ua = q[active, i], target[active, i]
        ptp = float(np.ptp(ua))
        metrics.append(dict(
            joint=name, excited=name in plan["excited"],
            target_ptp_rad=ptp, measured_ptp_rad=float(np.ptp(qa)),
            ptp_ratio=float(np.ptp(qa) / ptp) if ptp > 1e-12 else None,
            tracking_rmse_rad=float(np.sqrt(np.mean((qa - ua) ** 2))),
            mean_tracking_error_rad=float(np.mean(qa - ua)),
            max_abs_tracking_error_rad=float(np.max(np.abs(qa - ua))),
            max_abs_velocity_rad_s=float(np.max(np.abs(vel[:, i]))),
            max_abs_feedback_torque_Nm=float(np.max(np.abs(torque[:, i]))),
            max_abs_nominal_pd_Nm=float(np.max(np.abs(pd_proxy[:, i]))),
            max_nominal_pd_over_manifest_limit=float(np.max(pd_ratio[:, i])),
        ))
    summary = dict(
        source=str(directory.resolve()), status=status, group=plan["group"],
        checks=checks, basic_integrity_pass=all(checks.values()),
        raw_rows=len(raw), sweep_samples=len(sweep), expected_sweep_samples=plan["sample_count"],
        sweep_recorded_duration_s=len(sweep) * dt, last_sweep_time_s=float(time[-1]),
        expected_sweep_duration_s=plan["sample_count"] * dt,
        interval_min_max_ms=[float(interval.min() * 1000), float(interval.max() * 1000)],
        max_interval_jitter_us=float(np.max(np.abs(interval - dt)) * 1e6),
        max_grid_jitter_us=float(np.max(np.abs(grid_error)) * 1e6),
        fixed_joint_max_abs_rad=fixed_max,
        peak_logged_nominal_pd=dict(joint=names[peak[1]], time_s=float(time[peak[0]]),
            value_Nm=float(pd_proxy[peak]), limit_Nm=float(effort[peak[1]]), ratio=float(pd_ratio[peak])),
        last_logged_nominal_pd=dict(joint=names[last], value_Nm=float(pd_proxy[-1, last]),
            limit_Nm=float(effort[last]), ratio=float(pd_ratio[-1, last])),
    )
    gyro_cols = [f"imu_w{axis}" for axis in "xyz"]
    if all(c in col for c in gyro_cols):
        gyro = sweep[:, [col[c] for c in gyro_cols]]
        summary["imu_gyro_rms_deg_s_xyz"] = np.rad2deg(np.sqrt(np.mean(gyro ** 2, axis=0))).tolist()
        summary["imu_gyro_peak_deg_s_xyz"] = np.rad2deg(np.max(np.abs(gyro), axis=0)).tolist()
    out = directory / "plots"
    out.mkdir(exist_ok=True)
    with (out / "joint_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    shown = names if all_joints else [n for n in names if n in plan["excited"]]
    indices = [names.index(n) for n in shown]
    if not indices:
        raise ValueError(f"{directory}: no joints selected")
    end = plan["sample_count"] * dt
    interrupted = status["result"] != "complete" or not complete_samples
    for kind, data, units, label in (
        ("position", q, "Pos [rad]", "pos"),
        ("torque", torque, "Torque [Nm]", "SDK feedback torque"),
        ("velocity", vel, "Vel [rad/s]", "vel"),
    ):
        rows = (len(indices) + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 3), sharex=True, squeeze=False)
        for ax, i in zip(axes.flat, indices):
            ax.plot(time, data[:, i], label=label, linewidth=.8)
            if kind == "position":
                ax.plot(time, target[:, i], "--", label="target", linewidth=.9)
            ax.axvspan(0, min(plan["rest"], time[-1]), color="gray", alpha=.12)
            if time[-1] > plan["rest"] + plan["duration"]:
                ax.axvspan(plan["rest"] + plan["duration"], time[-1], color="gray", alpha=.12)
            if interrupted:
                ax.axvspan(time[-1], end, color="tab:red", alpha=.07)
                ax.axvline(time[-1], color="tab:red", linestyle=":", linewidth=.8)
            ax.set_xlim(0, end)
            ax.set_title(names[i], fontsize=9)
            ax.set_xlabel("Phase-2 time [s]", fontsize=8)
            ax.set_ylabel(units, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.3)
            ax.legend(fontsize=7, loc="upper left")
        for ax in list(axes.flat)[len(indices):]:
            ax.set_visible(False)
        fig.suptitle(f"S800 Real Chirp - {plan['group']} - {kind}\n{directory.name}\n{status['result']}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(out / f"{kind}.png", dpi=150)
        plt.close(fig)
    report = [f"# 真机采集诊断：{directory.name}", "",
        f"- 采集状态：`{status['result']}`；激励组：`{plan['group']}`。",
        f"- 扫频阶段记录 {len(sweep)}/{plan['sample_count']} 行，约 {len(sweep)*dt:.3f}/{end:.3f} 秒（含 rest）。",
        f"- 采样间隔范围：{interval.min()*1000:.6f}～{interval.max()*1000:.6f} ms。",
        "- 图默认只显示被激励关节；joint_metrics.csv 同时检查未激励的 22 自由度模型关节。",
        "- 图保留 phase=2 的前后 rest，排除入场和增益切换；灰色为 rest，红色空白为未采集部分。",
        "", "## 数据完整性", ""]
    report.extend(f"- `{key}`: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
    report += ["", "## 激励期间跟踪指标", "",
        "| 关节 | 目标峰峰值 rad | 实际峰峰值 rad | RMSE rad | 平均误差 rad | 最大速度 rad/s | 最大反馈力矩 Nm |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        if m["excited"]:
            report.append(f"| {m['joint']} | {m['target_ptp_rad']:.5f} | {m['measured_ptp_rad']:.5f} | {m['tracking_rmse_rad']:.5f} | {m['mean_tracking_error_rad']:.5f} | {m['max_abs_velocity_rad_s']:.4f} | {m['max_abs_feedback_torque_Nm']:.3f} |")
    report += ["", "## 解释边界", "",
        "- 位置误差采用同一行的 q − q_des；状态在该行命令发布前读取，不擅自移位。平均误差不等于编码器 bias。",
        "- 峰峰值比不是频响增益；中断记录的指标只覆盖已采集区间，不可与完整扫频直接比较。",
        "- 力矩图仅画 SDK 关节反馈。没有真实摩擦参数，无法复现仿真图的 net_torque；也不将反馈力矩等同于已校准的施加力矩。",
        "- nominal PD = kp*(q_des-q)-kd*qd，仅用于诊断，不能替代真实电机命令或证明并联机构的映射正确。",
        "- PD 保护在写入该周期数据前触发；最后一行是已记录的前一周期，无法仅凭它确认触发关节。summary 中 PD 排名只覆盖模型的 22 个关节，保护检查覆盖整机。",
        "- 完整性通过不代表已满足固定基座、无碰撞、充分激励等辨识条件；这里没有启动仿真或导出拟合数据。",
        ""]
    (out / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Single recording directory or its parent")
    parser.add_argument("--all-joints", action="store_true", help="Plot all 22 model joints, including held joints")
    parser.add_argument("--all-joint-torque", action="store_true",
                        help="Only generate full-recording torque PNG/PDF/peak CSV for all 43 SDK joints")
    args = parser.parse_args()
    directories = [args.path] if (args.path / "raw.csv").is_file() else sorted(p.parent for p in args.path.glob("*/raw.csv"))
    if not directories:
        parser.error("No raw.csv recordings found")
    failed = []
    for directory in directories:
        try:
            if args.all_joint_torque:
                plot_all_joint_torque(directory)
            else:
                analyze(directory, args.all_joints)
        except (ValueError, KeyError, OSError) as exc:
            failed.append(f"{directory}: {exc}")
    if failed:
        raise SystemExit("\n".join(failed))


if __name__ == "__main__":
    main()
