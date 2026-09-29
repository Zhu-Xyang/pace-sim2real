# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

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
# Parameter comparison table: identified vs. ground-truth (preset) values
# ----------------------------------------------------------------------------
# Ground-truth values used in data_collection.py to generate the "real" data.
# These must be kept in sync with scripts/pace/data_collection.py.
GT_ARMATURE = {
    "J00_HIP_PITCH_L": 0.24, "J01_HIP_ROLL_L": 0.14, "J02_HIP_YAW_L": 0.05,
    "J03_KNEE_PITCH_L": 0.24, "J04_ANKLE_PITCH_L": 0.05, "J05_ANKLE_ROLL_L": 0.05,
    "J06_HIP_PITCH_R": 0.24, "J07_HIP_ROLL_R": 0.14, "J08_HIP_YAW_R": 0.05,
    "J09_KNEE_PITCH_R": 0.24, "J10_ANKLE_PITCH_R": 0.05, "J11_ANKLE_ROLL_R": 0.05,
    "J12_TORSO_YAW": 0.05,
    "J13_SHOULDER_PITCH_L": 0.05, "J14_SHOULDER_ROLL_L": 0.05, "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.05, "J17_ELBOW_YAW_L": 0.008, "J18_WRIST_PITCH_L": 0.008, "J19_WRIST_ROLL_L": 0.008,
    "J27_SHOULDER_PITCH_R": 0.05, "J28_SHOULDER_ROLL_R": 0.05, "J29_SHOULDER_YAW_R": 0.05,
    "J30_ELBOW_PITCH_R": 0.05, "J31_ELBOW_YAW_R": 0.008, "J32_WRIST_PITCH_R": 0.008, "J33_WRIST_ROLL_R": 0.008,
}

GT_VISCOUS = {
    "J00_HIP_PITCH_L": 1.6, "J01_HIP_ROLL_L": 1.6, "J02_HIP_YAW_L": 1.0,
    "J03_KNEE_PITCH_L": 1.6, "J04_ANKLE_PITCH_L": 0.5, "J05_ANKLE_ROLL_L": 0.5,
    "J06_HIP_PITCH_R": 1.6, "J07_HIP_ROLL_R": 1.6, "J08_HIP_YAW_R": 1.0,
    "J09_KNEE_PITCH_R": 1.6, "J10_ANKLE_PITCH_R": 0.5, "J11_ANKLE_ROLL_R": 0.5,
    "J12_TORSO_YAW": 1.0,
    "J13_SHOULDER_PITCH_L": 0.5, "J14_SHOULDER_ROLL_L": 0.5, "J15_SHOULDER_YAW_L": 0.5,
    "J16_ELBOW_PITCH_L": 0.5, "J17_ELBOW_YAW_L": 0.1, "J18_WRIST_PITCH_L": 0.1, "J19_WRIST_ROLL_L": 0.1,
    "J27_SHOULDER_PITCH_R": 0.5, "J28_SHOULDER_ROLL_R": 0.5, "J29_SHOULDER_YAW_R": 0.5,
    "J30_ELBOW_PITCH_R": 0.5, "J31_ELBOW_YAW_R": 0.1, "J32_WRIST_PITCH_R": 0.1, "J33_WRIST_ROLL_R": 0.1,
}

GT_FRICTION = {
    "J00_HIP_PITCH_L": 1.0, "J01_HIP_ROLL_L": 1.0, "J02_HIP_YAW_L": 0.3,
    "J03_KNEE_PITCH_L": 1.0, "J04_ANKLE_PITCH_L": 0.3, "J05_ANKLE_ROLL_L": 0.3,
    "J06_HIP_PITCH_R": 1.0, "J07_HIP_ROLL_R": 1.0, "J08_HIP_YAW_R": 0.3,
    "J09_KNEE_PITCH_R": 1.0, "J10_ANKLE_PITCH_R": 0.3, "J11_ANKLE_ROLL_R": 0.3,
    "J12_TORSO_YAW": 0.3,
    "J13_SHOULDER_PITCH_L": 0.3, "J14_SHOULDER_ROLL_L": 0.3, "J15_SHOULDER_YAW_L": 0.3,
    "J16_ELBOW_PITCH_L": 0.3, "J17_ELBOW_YAW_L": 0.05, "J18_WRIST_PITCH_L": 0.05, "J19_WRIST_ROLL_L": 0.05,
    "J27_SHOULDER_PITCH_R": 0.3, "J28_SHOULDER_ROLL_R": 0.3, "J29_SHOULDER_YAW_R": 0.3,
    "J30_ELBOW_PITCH_R": 0.3, "J31_ELBOW_YAW_R": 0.05, "J32_WRIST_PITCH_R": 0.05, "J33_WRIST_ROLL_R": 0.05,
}


# GT_VISCOUS = 1.6    # all joints
# GT_FRICTION = 0.2   # all joints
GT_BIAS = 0.05      # all joints
GT_DELAY = 5        # sim steps

if plot_table:
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

    print("\n" + sep)
    print(header)
    print(sep)

    csv_rows = []

    for i, name in enumerate(joint_order):
        # Ground truth
        a_gt = GT_ARMATURE.get(name, 0.0)
        v_gt = GT_VISCOUS.get(name, 0.0)
        f_gt = GT_FRICTION.get(name, 0.0)
        b_gt = GT_BIAS

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
            f"  {v_gt:>9.1f} {v_id:>9.4f} {v_err_str}"
            f"  {f_gt:>9.1f} {f_id:>9.4f} {f_err_str}"
            f"  {b_gt:>9.2f} {b_id:>9.4f} {b_err_str}"
        )

        csv_rows.append([name, a_gt, a_id, a_err, v_gt, v_id, v_err, f_gt, f_id, f_err, b_gt, b_id, b_err])

    print(sep)

    # Delay row
    d_err = (id_delay - GT_DELAY) / GT_DELAY * 100 if GT_DELAY != 0 else 0.0
    d_err_padded = f"{d_err:+.1f}%"
    if abs(d_err) < 50.0:
        d_err_str = f"{GREEN}{d_err_padded}{RESET}"
    else:
        d_err_str = f"{RED}{d_err_padded}{RESET}"
    print(f"Delay: GT={GT_DELAY}, ID={id_delay:.4f}, Err={d_err_str}")

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
        writer.writerow(["Delay", GT_DELAY, id_delay, f"{d_err:.1f}"])
    print(f"\nComparison table saved to: {csv_path}")

print("Plotting complete.")
