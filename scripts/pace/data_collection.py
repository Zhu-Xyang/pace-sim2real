# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=8.0, help="Maximum frequency for the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from torch import pi

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root

def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device, dtype=torch.int32)

    # print joints order
    print("Isaac Lab joint names:", articulation.joint_names)
    print("Config joint_order:", joint_order)
    print("Mapped joint_ids:", joint_ids)

    # 修改为（S800 27 关节，使用 XML 真实值）：
    armature = torch.tensor([
        0.2427264, 0.14110848, 0.0448737, 0.2427264, 0.0354625, 0.0354625,  # 左腿
        0.2427264, 0.14110848, 0.0448737, 0.2427264, 0.0354625, 0.0354625,  # 右腿
        0.0448737,                                                             # 腰
        0.0354625, 0.0354625, 0.0354625, 0.0354625, 0.00671625, 0.005, 0.005, # 左臂
        0.0354625, 0.0354625, 0.0354625, 0.0354625, 0.00671625, 0.005, 0.005, # 右臂
    ], device=env.unwrapped.device).unsqueeze(0)

    # viscous friction (d in PACE Eq.6) — per motor type, must match s800_pace_env_cfg.py
    #   7520-22 (HIP_PITCH/ROLL/KNEE): 1.6,  7520-14 (HIP_YAW/TORSO): 1.0,
    #   5020 (ANKLE/SHOULDER/ELBOW_PITCH): 0.5,  4010 (ELBOW_YAW/WRIST): 0.1
    damping = torch.tensor([
        1.6, 1.6, 1.0, 1.6, 0.5, 0.5,  # 左腿: HIP_PITCH, HIP_ROLL, HIP_YAW, KNEE, ANKLE_P, ANKLE_R
        1.6, 1.6, 1.0, 1.6, 0.5, 0.5,  # 右腿
        1.0,                             # 腰: TORSO_YAW
        0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1,  # 左臂: SHOULDER_P/R/Y, ELBOW_P, ELBOW_Y, WRIST_P/R
        0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1,  # 右臂
    ], device=env.unwrapped.device).unsqueeze(0)

    friction = torch.tensor([1.5] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)  # coulomb friction

    bias = torch.tensor([0.05] * 27, device=env.unwrapped.device).unsqueeze(0)

    time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)

    env.reset()

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature), device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping), device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_viscous_friction_coeff[:, joint_ids] = damping
    # note: modeling coulomb friction if joint_friction = joint_dynamic_friction
    # If we set static friction lower than dynamic friction, the sim complains. So we need to do this weird order.
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0], device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_dynamic_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0], device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_friction_coeff[:, joint_ids] = friction
    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.int32))

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

    # Create a chirp signal for each action dimension

    duration = args_cli.duration  # seconds
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()  # Hz
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    f0 = args_cli.min_frequency  # Hz
    f1 = args_cli.max_frequency  # Hz

    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
    chirp_signal = torch.sin(phase)

    trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)
    # ===========anymal quadrupedal============
    # for anymal quadrupedal robot
    # trajectory_directions = torch.tensor(
    #     [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    #     device=env.unwrapped.device
    # )
    # trajectory_bias = torch.tensor(
    #     [0.0, 0.4, 0.8] * 4,
    #     device=env.unwrapped.device
    # )
    # trajectory_scale = torch.tensor(
    #     [0.25, 0.5, -2.0] * 4,
    #     device=env.unwrapped.device
    # )
    # ==============================

    # ===========s800==============
    trajectory_directions = torch.tensor(
        [  1,  1,  1,  1,  1,  1,      # 左腿 (轴相同 → dir=+1)
        1, -1, -1, -1, -1, -1,     # 右腿 (HIP_PITCH轴镜像→dir=+1, 其余→dir=-1)
        1,                          # 腰
        1,  1,  1,  1,  1,  1,  1, # 左臂
        1, -1,  1,  1,  1,  1,  1   # 右臂 (SHOULDER_PITCH/YAW, ELBOW×2, WRIST×2 轴镜像→dir=+1)
        ],
        device=env.unwrapped.device
    )

    # bias: 0 for symmetric joints, URDF center for asymmetric (knee/ankle_roll/shldr_roll/elbow_pitch/wrist_roll)
    trajectory_bias = torch.tensor(
        [0.000, 0.150, 0.000, 1.046, 0.000, -0.087,  # 左腿
        0.000, 0.150, 0.000, -1.046, 0.000, -0.087,  # 右腿 (与左腿相同!)
        0.000,                                        # 腰
        0.000, 1.052, 0.000, -1.004, 0.000, 0.000, -0.131,  # 左臂
        0.000, 1.052, 0.000, -1.004, 0.000, 0.000, -0.131], # 右臂 (与左臂相同!)
        device=env.unwrapped.device
    )

    # trajectory_bias = torch.tensor(
    #     [0.000, 0.000, 0.000, 1.046, 0.000, -0.087,  # 左腿
    #      0.000, 0.000, 0.000, -1.046, 0.000,  0.087,  # 右腿
    #      0.000,                                        # 腰
    #      0.000, 1.052, 0.000, -1.004, 0.000, 0.000, -0.131,  # 左臂
    #      0.000, 1.052, 0.000, 1.004, 0.000, 0.000,  0.131],  # 右臂
    #     device=env.unwrapped.device
    # )

    # scale: min(half_range * 0.4, 80% torque limit, phys_cap)
    trajectory_scale = torch.tensor(
        [0.500, 0.300, 0.700, 0.500, 0.272, 0.105,  # 左腿
         0.500, 0.300, 0.700, 0.500, 0.272, 0.105,  # 右腿
         0.400,                                       # 腰
         0.600, 0.400, 0.400, 0.400, 0.40, 0.05, 0.05,  # 左臂 (WRIST 0.25→0.10 避免力矩饱和)
         0.600, 0.400, 0.400, 0.400, 0.40, 0.05, 0.05],  # 右臂
        device=env.unwrapped.device
    )
    # ===========================

    trajectory[:, joint_ids] = (trajectory[:, joint_ids] + trajectory_bias.unsqueeze(0)) * trajectory_directions.unsqueeze(0) * trajectory_scale.unsqueeze(0)

    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias[0, joint_ids])
    articulation.write_joint_velocity_to_sim(torch.zeros((1, len(joint_ids)), device=env.unwrapped.device))

    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_vel_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_torque_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    time_data = t
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            robot = env.unwrapped.scene.articulations["robot"]
            dof_pos_buffer[counter, :] = robot.data.joint_pos[0, joint_ids] - bias[0]
            dof_vel_buffer[counter, :] = robot.data.joint_vel[0, joint_ids]
            dof_torque_buffer[counter, :] = robot.data.applied_torque[0, joint_ids]
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            dof_target_pos_buffer[counter, :] = robot._data.joint_pos_target[0, joint_ids]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter/sample_rate} seconds")
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

    from time import sleep
    sleep(1)  # wait a bit for everything to settle

    (data_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "time": time_data.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
        "dof_vel": dof_vel_buffer.cpu(),
        "dof_torque": dof_torque_buffer.cpu(),
    }, data_dir / "chirp_data.pt")

    import matplotlib.pyplot as plt

    n_joints = len(joint_ids)
    n_cols = 4
    n_rows = (n_joints + n_cols - 1) // n_cols

    # ── Compute friction torque and net torque ──
    # friction_torque = viscous_friction * q̇ + coulomb_friction * sign(q̇)
    # net_torque = applied_torque - friction_torque
    viscous = damping[0].cpu()  # (n_joints,)
    coulomb = friction[0].cpu()  # (n_joints,)
    vel_cpu = dof_vel_buffer.cpu()  # (num_steps, n_joints)
    torque_cpu = dof_torque_buffer.cpu()  # (num_steps, n_joints)
    friction_torque = viscous.unsqueeze(0) * vel_cpu + coulomb.unsqueeze(0) * torch.sign(vel_cpu)
    net_torque = torque_cpu - friction_torque

    # ── Figure 1: joint position ──
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True)
    axes_flat = axes.flatten() if n_joints > 1 else [axes]

    for i in range(n_joints):
        ax = axes_flat[i]
        ax.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label="pos")
        ax.plot(t.cpu().numpy(), dof_target_pos_buffer[:, i].cpu().numpy(), label="target", linestyle='dashed')
        ax.set_title(joint_order[i], fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Pos [rad]", fontsize=8)
        ax.grid(True)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for i in range(n_joints, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("S800 Chirp Trajectory — All Joints", fontsize=12)
    plt.tight_layout()

    # ── Figure 2: applied torque & net torque (after friction) ──
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True)
    axes2_flat = axes2.flatten() if n_joints > 1 else [axes2]

    for i in range(n_joints):
        ax = axes2_flat[i]
        ax.plot(t.cpu().numpy(), torque_cpu[:, i].numpy(), label="applied_torque", linewidth=0.8)
        ax.plot(t.cpu().numpy(), net_torque[:, i].numpy(), label="net_torque", linewidth=0.8, linestyle='dashed')
        ax.set_title(joint_order[i], fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Torque [Nm]", fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")

    for i in range(n_joints, len(axes2_flat)):
        axes2_flat[i].set_visible(False)

    fig2.suptitle("S800 Chirp — Applied Torque vs Net Torque (after friction)", fontsize=12)
    plt.tight_layout()

    # ── Figure 3: joint velocity ──
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True)
    axes3_flat = axes3.flatten() if n_joints > 1 else [axes3]

    for i in range(n_joints):
        ax = axes3_flat[i]
        ax.plot(t.cpu().numpy(), vel_cpu[:, i].numpy(), label="vel", color="tab:blue", linewidth=0.8)
        ax.set_title(joint_order[i], fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Vel [rad/s]", fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    for i in range(n_joints, len(axes3_flat)):
        axes3_flat[i].set_visible(False)

    fig3.suptitle("S800 Chirp — Joint Velocity", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
