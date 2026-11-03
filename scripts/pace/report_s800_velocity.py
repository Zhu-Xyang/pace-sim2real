"""Offline S800 chirp velocity report; does not launch Isaac Sim or send commands."""

import argparse
import ast
import csv
import importlib.util
import math
from pathlib import Path

import torch

from experiment_utils import group_chirp
from s800_motion import scaled_nominal_motion

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/s800_pace_env_cfg.py"


def source_settings():
    """Read constants without importing the simulator or executing the collector."""
    tree = ast.parse(CFG.read_text())
    names = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                 and n.target.id == "joint_order")
    dt = next(ast.literal_eval(n.value) for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and any(ast.unparse(t) == "self.sim.dt" for t in n.targets))
    defaults = {}
    for n in ast.walk(ast.parse((ROOT / "scripts/pace/data_collection.py").read_text())):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)):
            for kw in n.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[n.args[0].value] = kw.value.value
    return names, dt, defaults


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(c) for c in r) + " |" for r in rows],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/reports/s800_velocity_limits")
    parser.add_argument("--report-date", default="2026-10-29")
    parser.add_argument("--legs-factor", type=float, default=0.8)
    parser.add_argument("--arms-factor", type=float, default=1.0)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("s800_kpkd", ROOT / "assets_robot/kpkd_params.py")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    active, dt, defaults = source_settings()
    settings = {"dt": dt, "duration": defaults["--duration"], "rest": defaults["--rest"],
                "ramp": defaults["--ramp"], "f0": defaults["--min_frequency"]}
    group_names = {
        "legs": [n for n in active if any(x in n for x in ("HIP_", "KNEE_", "ANKLE_"))],
        "arms": [n for n in active if any(x in n for x in ("SHOULDER_", "ELBOW_", "WRIST_"))],
    }
    if set(sum(group_names.values(), [])) != set(active):
        raise ValueError("Active joint groups changed; review report assumptions before regenerating")
    results = {}
    for group, names in group_names.items():
        factor = args.legs_factor if group == "legs" else args.arms_factor
        selected = list(range(len(names)))
        center, amplitude = scaled_nominal_motion(names, selected, factor)
        peaks = {}
        for frequency in (4, 8):
            _, commands = group_chirp(**settings, f1=torch.full((len(names),), float(frequency)),
                                      center=center, amplitude=amplitude, selected=selected)
            # Mean segment velocity of the sampled target, not measured robot speed.
            peaks[frequency] = commands.diff(dim=0).abs().amax(dim=0) / dt
        for j, name in enumerate(names):
            limit = float(params.S800_VELOCITY_LIMITS[name])
            a = abs(float(amplitude[j]))
            row = dict(joint=name, group=group, active=True, velocity_limit_rad_s=limit,
                       kpkd_action_scale_rad=float(params.S800_HYBRID_ACTION_SCALE[name]),
                       scale_factor=factor, chirp_amplitude_rad=a,
                       screening_frequency_hz=limit / (2 * math.pi * a))
            for f in (4, 8):
                estimate = 2 * math.pi * f * a
                sampled = float(peaks[f][j])
                row.update({f"estimate_{f}hz_rad_s": estimate,
                            f"sampled_{f}hz_rad_s": sampled,
                            f"sampled_{f}hz_utilization_pct": sampled / limit * 100,
                            f"sampled_{f}hz_exceeds_limit": sampled > limit})
            results[name] = row
    rows = []
    for name in params.S800_JOINT_NAMES:
        if name in results:
            rows.append(results[name])
        else:
            rows.append(dict(joint=name, group="locked", active=False,
                             velocity_limit_rad_s=float(params.S800_VELOCITY_LIMITS[name]),
                             kpkd_action_scale_rad=float(params.S800_HYBRID_ACTION_SCALE[name])))
    fields = list(next(iter(results.values())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "joint_velocity.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# S800 逐关节速度限制报告：4 Hz 与 8 Hz",
        f"报告日期：{args.report_date}。依据生成时本地配置计算；本报告没有启动仿真或真机。",
        "## 1. 数据来源与计算条件",
        "- 速度限值：`assets_robot/kpkd_params.py` 的 `S800_VELOCITY_LIMITS`，即 `_TYPE_TABLE` 的 `v_lim` 列。单位为关节输出侧 rad/s，不能当作电机转子 rpm。\n"
        "- 扫频幅值：`scripts/pace/s800_motion.py` 的 `nominal_motion` / `scaled_nominal_motion`。`kpkd_params.py` 的 `S800_HYBRID_ACTION_SCALE` 并非采集器当前使用的 chirp 幅值。\n"
        "- 活动关节与周期：`s800_pace_env_cfg.py` 的 `joint_order` 与 `self.sim.dt`。\n"
        "- 波形：直接调用 `experiment_utils.py` 的 `group_chirp`；频率、静止和淡入淡出时间的默认值从 `data_collection.py` 读取。",
        f"主表对应分别执行 `--group legs` 和 `--group arms`，采用 `--trajectory_profile scaled_legacy`，"
        f"腿部幅值乘 {args.legs_factor:g}，手臂幅值乘 {args.arms_factor:g}。两组的结果合并展示，不代表同时激励全身。"
        "未使用 `--grouped_sweep`、`--trajectory_plan` 或 `clearance`；`--group all` 默认连手臂也乘 0.8，与本报告不同。",
        f"采样周期 {dt:g} s（{1/dt:g} Hz）；起始频率 {settings['f0']:g} Hz；扫频 {settings['duration']:g} s；"
        f"前后各静止 {settings['rest']:g} s；首尾各 {settings['ramp']:g} s 余弦淡入淡出。分别将统一扫频上限设为 4 Hz、8 Hz。",
        "本报告采用的 `kpkd_params.py` 限值与其 27 个关节在原始 S800 URDF 中的 `limit velocity` 一致。"
        "这只是模型配置值，不代表已核实的真机试验允许速度，尤其腕部 150 rad/s 需要核对实际驱动器与厂商资料。",
        "## 2. 指标含义",
        "- **幅值 A**：最终正弦单边幅值，单位 rad；峰峰值为 2A。常数中心偏置不改变稳态导数。\n"
        "- **估算峰值**：`2π × f_max × A`，是不考虑窗函数导数的快速筛查值，不是所有带窗轨迹的严格上界。\n"
        "- **采样峰值**：`max(abs(q_target[k+1] − q_target[k])) / dt`，用当前代码生成的 float32 目标位置计算。它是相邻采样段的平均速度峰值，不是真机速度、仿真响应或连续时间峰值的证明。\n"
        "- **占用率**：采样峰值 / 配置速度限值 × 100%；超过 100% 标记为“超限”。未超过仅说明这项离散目标检查未超限。",
        "当前波形满足 `q(t)=c+A·w(t)·sin(φ(t))`，其连续导数为 "
        "`dq/dt=A·[dw/dt·sin(φ)+2πf(t)·w(t)·cos(φ)]`。因此正式执行前还应结合控制器实际插值/保持方式检查轨迹，并监测实际关节速度。",
    ]
    for group, label in (("legs", "腿部（12 关节）"), ("arms", "手臂（10 关节）")):
        report.append(f"## {'3' if group == 'legs' else '4'}. {label}")
        report.append("所有速度单位为 rad/s；两组频率的占用率均按采样峰值计算。")
        report.append(table(
            ["关节", "限值", "A (rad)", "4 Hz 估算", "4 Hz 采样", "4 Hz 占用", "8 Hz 估算", "8 Hz 采样", "8 Hz 占用", "检查结果"],
            [[r["joint"], f"{r['velocity_limit_rad_s']:.2f}", f"{r['chirp_amplitude_rad']:.4f}",
              f"{r['estimate_4hz_rad_s']:.2f}", f"{r['sampled_4hz_rad_s']:.2f}", f"{r['sampled_4hz_utilization_pct']:.1f}%",
              f"{r['estimate_8hz_rad_s']:.2f}", f"{r['sampled_8hz_rad_s']:.2f}", f"{r['sampled_8hz_utilization_pct']:.1f}%",
              "**8 Hz 超限**" if r['sampled_8hz_exceeds_limit'] else "两项未超限"]
             for r in rows if r['group'] == group]))
    report += ["## 5. 已锁定关节（保留配置限值，不参与当前扫频）",
               "以下关节不在当前 `joint_order` 中，已在锁定版 URDF 中设为 fixed。主报告中的速度值记为“不适用”，不能将未运行解释为通过速度验收。",
               table(["关节", "配置限值 rad/s", "4 Hz / 8 Hz"],
                     [[r['joint'], f"{r['velocity_limit_rad_s']:.2f}", "不适用：已锁定"] for r in rows if not r['active']]),
               "## 6. 结论与使用边界"]
    for f in (4, 8):
        bad = [r['joint'] for r in rows if r['active'] and r[f'sampled_{f}hz_exceeds_limit']]
        report.append(f"- **{f} Hz**：22 个活动关节中，{len(bad)} 个目标采样速度超限。" + ("关节：" + "、".join(bad) + "。" if bad else ""))
    yaw = results['J02_HIP_YAW_L']
    report += [
        f"左右髋偏航是本设定中的主要速度约束：A={yaw['chirp_amplitude_rad']:.2f} rad，"
        f"8 Hz 采样峰值 {yaw['sampled_8hz_rad_s']:.2f} rad/s，对应限值 {yaw['velocity_limit_rad_s']:.2f} rad/s 的 "
        f"{yaw['sampled_8hz_utilization_pct']:.1f}%。",
        f"按简式反推，髋偏航在当前幅值下的频率筛查边界为 "
        f"`v_lim/(2πA)≈{yaw['screening_frequency_hz']:.2f} Hz`；保持 8 Hz 时，幅值筛查边界为 "
        f"`v_lim/(2π×8)≈{yaw['velocity_limit_rad_s']/(2*math.pi*8):.3f} rad`。"
        "这些只是忽略窗导数、未留实验裕度的筛查数值，不是推荐真机运行值。调整后应重新生成完整轨迹检查。",
        "速度合格不代表轨迹可执行：加速度、负载下的力矩—转速边界、位置限位、接触、控制延迟及实际跟踪振荡仍会影响运动。"
        "当前采集脚本没有逐关节目标速度超限拒绝机制；本报告没有修改采集器或硬件保护。",
        "## 7. 附录：若直接把 kpkd 的 action scale 当作正弦幅值",
        "本表仅回答另一种常见计算口径：假设 `A=S800_HYBRID_ACTION_SCALE`、归一化正弦幅值为 1，"
        "计算 `2πfA`。**这不是当前采集器轨迹**，也没有应用分组缩放、窗函数或锁定。"
        "包含全部 27 个配置关节，已锁定关节仅作假设算例。",
        table(["关节", "A=action scale", "限值 rad/s", "4 Hz 估算 rad/s", "8 Hz 估算 rad/s", "8 Hz 估算占用"],
              [[r['joint'], f"{r['kpkd_action_scale_rad']:.2f}", f"{r['velocity_limit_rad_s']:.2f}",
                f"{2*math.pi*4*r['kpkd_action_scale_rad']:.2f}", f"{2*math.pi*8*r['kpkd_action_scale_rad']:.2f}",
                f"{2*math.pi*8*r['kpkd_action_scale_rad']/r['velocity_limit_rad_s']*100:.1f}%"] for r in rows]),
        "## 8. 文件与复现",
        "- [逐关节主表 CSV](joint_velocity.csv)：27 行，含 22 个活动关节与 5 个锁定关节；锁定关节的轨迹数值留空。\n"
        "- [action scale 假设口径 CSV](action_scale_estimate.csv)：27 行，独立保存附录算例。\n"
        "- 脚本：`scripts/pace/report_s800_velocity.py`；无需启动 Isaac Sim。",
        "```bash\npython scripts/pace/report_s800_velocity.py --report-date " + args.report_date + "\n```",
        "脚本从当前源码重新读取限值、幅值、活动关节和时间参数；源码变动后重新生成会更新报告。"
        "`--legs-factor` 与 `--arms-factor` 默认分别为 0.8 和 1.0；如果采集时另行设置缩放，应传入相同数值。"
        "脚本用于本文指定的统一 4/8 Hz、scaled_legacy 分组口径，不读取既有 `.pt` 记录。",
    ]
    (args.output_dir / "report.md").write_text("\n\n".join(report) + "\n")
    with (args.output_dir / "action_scale_estimate.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["joint", "active_in_current_model", "assumed_amplitude_rad", "velocity_limit_rad_s",
                         "estimate_4hz_rad_s", "estimate_8hz_rad_s", "estimate_8hz_utilization_pct"])
        for r in rows:
            a, v = r['kpkd_action_scale_rad'], r['velocity_limit_rad_s']
            writer.writerow([r['joint'], r['active'], a, v, 2*math.pi*4*a, 2*math.pi*8*a, 2*math.pi*8*a/v*100])
    print(f"Report: {args.output_dir / 'report.md'}")
    for r in rows:
        if r['active']:
            print(f"{r['joint']:<25} limit={r['velocity_limit_rad_s']:6.2f} "
                  f"4Hz={r['sampled_4hz_rad_s']:6.2f} 8Hz={r['sampled_8hz_rad_s']:6.2f} "
                  f"8Hz utilization={r['sampled_8hz_utilization_pct']:5.1f}%")


if __name__ == "__main__":
    main()
