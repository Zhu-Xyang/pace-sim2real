# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import sys
import torch
import matplotlib.pyplot as plt
import re
from pathlib import Path
import csv

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--folder_name", type=str, default=None, help="Name of the folder to use.")
parser.add_argument("--mean_name", type=str, default=None, help="Name of the parameters file to use.")
parser.add_argument("--robot_name", type=str, default="s800_sim", help="Name of the robot.")
parser.add_argument("--gt_preset", type=str, default="auto", choices=("auto", "design", "uniform"),
                    help="出表的真值口径。auto=优先用 run 的 config.pt 里带的注入真值（新版 "
                         "data_collection 会写进去），没有才回落到硬编码的 S800 分段表；"
                         "design/uniform=强制用回落的 S800 表（只给 26_09_28~30 那批老 run 用）。")
parser.add_argument("--plot_trajectory", action="store_true", help="Whether to plot the trajectory.")
parser.add_argument("--plot_score", action="store_true", help="Whether to plot the score over iterations.")
parser.add_argument("--plot_table", action="store_true", help="Whether to print and save a parameter comparison table.")

args = parser.parse_args()
folder_name = args.folder_name
mean_name = args.mean_name
robot_name = args.robot_name
plot_trajectory = args.plot_trajectory
plot_score = args.plot_score
plot_table = args.plot_table

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent

# folder_name = "25_10_24_12-05-07"
log_dir = project_root / "logs" / "pace" / robot_name

if not log_dir.exists():
    raise FileNotFoundError(f"No logs for robot {robot_name} under {log_dir}")

_pattern = re.compile(r"^mean_(\d+)\.pt$")

def find_latest_params(root: Path):
    best = None  # tuple (int, Path)
    for p in root.rglob("mean_*.pt"):
        m = _pattern.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        if best is None or num > best[0]:
            best = (num, p)
    return None if best is None else best[1], best[0]

# if no folder_name given, pick the most recent run folder for the robot
if not folder_name:
    robot_dir = log_dir
    candidates = [p for p in robot_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {robot_dir}")
    latest_folder = max(candidates, key=lambda p: p.stat().st_mtime)
    folder_name = latest_folder.name

# now point log_dir at the chosen robot/run folder
log_dir = log_dir / folder_name

if mean_name is None:
    params_path, params_num = find_latest_params(log_dir)
else:
    params_path = log_dir / mean_name
    params_num = int(_pattern.match(mean_name).group(1))
    if not params_path.exists():
        raise FileNotFoundError(f"Given params file {params_path} does not exist")
if params_path is None:
    raise FileNotFoundError(f"No mean_*.pt files found under {log_dir}")
print(f"Latest params file: {params_path}")

mean = torch.load(params_path)
config = torch.load(log_dir / "config.pt")

joint_order = config["joint_order"]
trajectories = torch.load(log_dir / "best_trajectory.pt")  # time x joints
real_trajectories = config["dof_pos"]  # time x joints
target_trajectories = config["des_dof_pos"]  # time x joints
time = config["time"]  # time

print(f"Best parameter set: {mean}")
print(f"Armature params: {mean[:len(joint_order)]}")
print(f"Viscous friction params: {mean[len(joint_order):2 * len(joint_order)]}")
print(f"Static/dynamic friction params: {mean[2 * len(joint_order):3 * len(joint_order)]}")
print(f"Encoder bias params: {mean[3 * len(joint_order):4 * len(joint_order)]}")
print(f"Delay param: {mean[-1].item()}")
encoder_bias = mean[3 * len(joint_order):4 * len(joint_order)]  # extract encoder bias

if plot_score:
    try:
        progress = torch.load(log_dir / "progress.pt")
        print("Loaded optimization progress.")
    except FileNotFoundError:
        progress = None
        print("No optimization progress file found. Skipping score plot.")
        plot_score = False

if plot_score:
    plt.figure()
    plt.title("CMA-ES Score over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    data = torch.min(progress["scores_buffer"][:params_num + 1], dim=1).values.cpu().numpy()
    plt.semilogy(data)
    plt.xlim(0, params_num)
    # plt.ylim(0, None)
    plt.grid()
    plt.show()

if plot_trajectory:
    for i in range(len(joint_order)):
        plt.figure(figsize=(8, 4.5))
        plt.plot(time, trajectories[:, i].cpu().numpy() - encoder_bias[i].item(), c="tab:orange", label="Sim", linewidth=2)  # in encoder frame
        plt.plot(time, real_trajectories[:, i].cpu().numpy(), label="Real", c="tab:green", linestyle="--", linewidth=2)
        plt.plot(time, target_trajectories[:, i].cpu().numpy(), c="grey", label="Target", linestyle="--", alpha=0.5)
        plt.title(f"Joint {joint_order[i]}")  # Use joint names from config
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

# ----------------------------------------------------------------------------
# Parameter comparison table: identified vs. ground truth
# ----------------------------------------------------------------------------
# 真值来源（优先级从高到低）：
#   1. run 自己的 config.pt 里带的「注入真值」—— data_collection.py 把它写进 chirp_data.pt，
#      fit 再转发到 config.pt。**新跑的 run 都走这条，永远和数据本身一致。**
#   2. 回落到 scripts/pace/robot_tables.py 里该机器人的注入真值表（单一数据源，和
#      data_collection 用的是同一份，不会漂移）。老 run（26_09_28~30）没有字段 1，走这条。
#   3. `--gt_preset uniform` 用对照组（全场统一值），只对 S800 有意义。
# ⚠️ 关节名对不上会**直接报错**，不会再静默显示 0% 误差（那个假完美踩过两次）。
sys.path.insert(0, str(Path(__file__).parent))
from robot_tables import JOINT_ORDER, ROBOT_TABLES, UNIFORM_CONTROL  # noqa: E402


def _resolve_gt(cfg, robot_key, preset):
    """返回 (armature, viscous, friction, bias 四张 {关节名: 值} 表, delay, 来源说明)。"""
    gt = cfg.get("gt") if isinstance(cfg, dict) else None
    if gt is not None and preset == "auto":
        order = [str(n) for n in gt["joint_order"]]
        if order != [str(n) for n in joint_order]:
            raise SystemExit(
                "[gt] ❌ config.pt 里记的 joint_order 与本 run 的 joint_order 不一致 —— "
                "这份 config 不是这个数据集的，不能用它出表。")
        mk = lambda key: {n: float(v) for n, v in zip(order, gt[key])}  # noqa: E731
        return (mk("armature"), mk("damping"), mk("friction"), mk("bias"),
                float(gt["delay"]), "config.pt 里记录的注入真值")

    table = ROBOT_TABLES.get(robot_key)
    order = JOINT_ORDER.get(robot_key)
    if table is None or order is None:
        raise SystemExit(
            f"[gt] ❌ robot_tables.py 里没有 {robot_key}（可用：{sorted(ROBOT_TABLES)}）。"
            f"用 --robot_name 指定机器人。")
    if sorted(order) != sorted(joint_order):
        raise SystemExit(
            f"[gt] ❌ robot_tables.py 里 {robot_key} 的关节列表与本 run 的 joint_order 对不上\n"
            f"     表里 {len(order)} 个，run 里 {len(joint_order)} 个。"
            f"数据可能是别的机器人采的，或表被改过。")

    if preset == "uniform":
        u = UNIFORM_CONTROL.get(robot_key)
        if u is None:
            raise SystemExit(f"[gt] ❌ {robot_key} 没有 uniform 对照组（只有 S800 做过）。")
        return ({n: u["armature"] for n in joint_order}, {n: u["viscous"] for n in joint_order},
                {n: u["friction"] for n in joint_order}, {n: u["bias"] for n in joint_order},
                u["delay"], f"--gt_preset uniform（全场 {u['armature']}/{u['viscous']}/{u['friction']}）")

    print("[gt] ⚠️ config.pt 里没有注入真值（老数据文件），回落到 robot_tables.py 的注入真值表。"
          "若这轮是用 --uniform_gt 采的，请加 --gt_preset uniform，否则算出来的误差是假的。")
    return (dict(zip(order, table["armature"])), dict(zip(order, table["damping"])),
            dict(zip(order, table["friction"])), {n: table["bias_gt"] for n in order},
            table["delay_gt"], f"robot_tables.py 的 {robot_key} 表（回落）")


if plot_table:
    gt_arm, gt_vis, gt_fri, gt_bias, gt_delay, gt_src = _resolve_gt(config, args.robot_name, args.gt_preset)
    _missing = [n for n in joint_order
                if n not in gt_arm or n not in gt_vis or n not in gt_fri or n not in gt_bias]
    if _missing:
        raise SystemExit(
            f"[gt] ❌ 有 {len(_missing)} 个关节在 GT 表里查不到：{_missing[:6]}"
            f"{' ...' if len(_missing) > 6 else ''}\n"
            f"     当前真值来源 = {gt_src}\n"
            f"     名字对不上时误差会静默变成 0%，所以这里直接退出而不是照常出表。\n"
            f"     新数据请用新版 data_collection.py 重采一次（注入真值会随数据落盘），"
            f"或用 --gt_preset 指定正确口径。")
    num_joints = len(joint_order)

    # Extract identified parameters from the mean vector
    id_armature = mean[0:num_joints]
    id_viscous = mean[num_joints:2 * num_joints]
    id_friction = mean[2 * num_joints:3 * num_joints]
    id_bias = mean[3 * num_joints:4 * num_joints]
    id_delay = mean[-1].item()

    # ANSI color codes for terminal output
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def color_err(err_str: str, err_val: float) -> str:
        """Wrap error string in green (<50%) or red (>=50%) ANSI color.
        Pad to 8 chars first so ANSI codes don't break alignment."""
        padded = f"{err_str:>8}"
        if abs(err_val) < 50.0:
            return f"{GREEN}{padded}{RESET}"
        else:
            return f"{RED}{padded}{RESET}"

    # Build table rows
    header = (
        f"{'Joint':<28}"
        f"{'Arm_GT':>10} {'Arm_ID':>10} {'Err%':>8}"
        f"  {'Visc_GT':>9} {'Visc_ID':>9} {'Err%':>8}"
        f"  {'Fric_GT':>9} {'Fric_ID':>9} {'Err%':>8}"
        f"  {'Bias_GT':>9} {'Bias_ID':>9} {'Err%':>8}"
    )
    sep = "=" * len(header)

    print(f"\n# 真值来源: {gt_src}   (--gt_preset={args.gt_preset})")
    # 哪些关节真的被拟合了。--group legs 这类分组轮里，其余关节的参数停在 bounds 中点，
    # 拿它们去和真值比会显示成一片巨大的误差 —— 那是「没拟合」，不是「拟合错了」。
    _fitted = config.get("fitted_joints")
    _fitted_set = set(_fitted) if _fitted is not None else None
    if _fitted_set is not None and len(_fitted_set) < len(joint_order):
        _names = [joint_order[i] for i in sorted(_fitted_set)]
        print(f"# 本轮只拟合了 {len(_fitted_set)}/{len(joint_order)} 个关节："
              f"{', '.join(_names[:6])}{' ...' if len(_names) > 6 else ''}")
        print("# 其余关节标 '·nf'（not fitted），它们的误差不代表拟合质量。")
    print(sep)
    print(header)
    print(sep)

    csv_rows = []

    for i, name in enumerate(joint_order):
        # Ground truth（来自 config.pt 的注入真值，或 --gt_preset 指定的回落表）
        a_gt = gt_arm[name]
        v_gt = gt_vis[name]
        f_gt = gt_fri[name]
        b_gt = gt_bias[name]

        # Identified
        a_id = id_armature[i].item()
        v_id = id_viscous[i].item()
        f_id = id_friction[i].item()
        b_id = id_bias[i].item()

        # Percentage errors
        a_err = (a_id - a_gt) / a_gt * 100 if a_gt != 0 else 0.0
        v_err = (v_id - v_gt) / v_gt * 100 if v_gt != 0 else 0.0
        f_err = (f_id - f_gt) / f_gt * 100 if f_gt != 0 else 0.0
        b_err = (b_id - b_gt) / b_gt * 100 if b_gt != 0 else 0.0

        # Color-coded error strings
        a_err_str = color_err(f"{a_err:>+7.1f}%", a_err)
        v_err_str = color_err(f"{v_err:>+7.1f}%", v_err)
        f_err_str = color_err(f"{f_err:>+7.1f}%", f_err)
        b_err_str = color_err(f"{b_err:>+7.1f}%", b_err)

        print(
            f"{name:<28}"
            f"{a_gt:>10.4f} {a_id:>10.4f} {a_err_str}"
            f"  {v_gt:>9.4f} {v_id:>9.4f} {v_err_str}"
            f"  {f_gt:>9.4f} {f_id:>9.4f} {f_err_str}"
            f"  {b_gt:>9.2f} {b_id:>9.4f} {b_err_str}"
            + ("   ·nf" if _fitted_set is not None and i not in _fitted_set else "")
        )

        csv_rows.append([name, a_gt, a_id, a_err, v_gt, v_id, v_err, f_gt, f_id, f_err, b_gt, b_id, b_err])

    print(sep)

    # Delay row
    d_err = (id_delay - gt_delay) / gt_delay * 100 if gt_delay != 0 else 0.0
    d_err_padded = f"{d_err:+.1f}%"
    if abs(d_err) < 50.0:
        d_err_str = f"{GREEN}{d_err_padded}{RESET}"
    else:
        d_err_str = f"{RED}{d_err_padded}{RESET}"
    print(f"Delay: GT={gt_delay:g}, ID={id_delay:.4f}, Err={d_err_str}")

    # Save CSV
    csv_path = log_dir / "param_comparison.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Joint",
            "Armature_GT", "Armature_ID", "Armature_Err%",
            "Viscous_GT", "Viscous_ID", "Viscous_Err%",
            "Friction_GT", "Friction_ID", "Friction_Err%",
            "Bias_GT", "Bias_ID", "Bias_Err%",
        ])
        writer.writerows(csv_rows)
        writer.writerow(["Delay", gt_delay, id_delay, f"{d_err:.1f}"])
    print(f"\nComparison table saved to: {csv_path}")

print("Plotting complete.")
