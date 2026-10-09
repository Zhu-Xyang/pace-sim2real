# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
from datetime import datetime
import json
from experiment_utils import (select_groups, recording_name, configure_delay, group_chirp, actuator_metadata)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=4.0, help="统一扫频上限 Hz；--grouped_sweep 开启时使用逐关节上限。")
parser.add_argument("--grouped_sweep", action=argparse.BooleanOptionalAction, default=False,
                    help="使用逐关节 3/6 Hz 上限，左右对应关节同相位；与 --group 的部位分组不同。")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
parser.add_argument("--group", choices=("all", "legs", "torso", "arms"), default="all",
                    help="只激励这一组，其余关节维持中心目标；arms 含手腕、不含手指。")
parser.add_argument("--delay_mode", choices=("command", "torque"), default=None,
                    help="分组实验默认 command；原全身实验默认 torque。")
parser.add_argument("--delay_steps", type=int, default=5, help="仿真注入的整数延时步数。")
parser.add_argument("--rest", type=float, default=1.0, help="扫频前后各静止多少秒。")
parser.add_argument("--ramp", type=float, default=2.0, help="扫频首尾各淡入淡出多少秒。")
parser.add_argument("--self_collisions", action=argparse.BooleanOptionalAction, default=None,
                    help="clearance 默认开启自碰撞；原轨迹两种 profile 分组默认关闭，全身沿用 cfg。")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="采集后默认显示位置、力矩、速度三张诊断图；--no-plot 可关闭。")
parser.add_argument("--output", type=str, default=None, help="输出路径，相对路径从 data/ 起算。")
parser.add_argument("--trajectory_profile", choices=("scaled_legacy", "clearance", "legacy"), default="scaled_legacy",
                    help="默认恢复原轨迹并降低 scale；clearance 为外展间隙方案，legacy 为未缩放原轨迹。")
parser.add_argument("--scale_factor", type=float, default=None,
                    help="仅 scaled_legacy 使用：本组原 scale 的乘数 (0,1]；legs/all 默认 0.8，上身组默认 1。")
parser.add_argument("--clearance_margin", type=float, default=0.02, help="腿间和离地最小保守间隙 [m]。")
parser.add_argument("--trajectory_plan", type=Path, default=None, help="读取之前保存的 trajectory.json 中的中心和幅值，仍须通过检查。")
parser.add_argument("--seed", type=int, default=0, help="仿真环境随机种子。")
# ── 对照组：全场统一 GT ──────────────────────────────────────────────────
# 目的：隔离「搜索维度」这一个变量。
# ANYmal 的 round-trip 能到 <1%，一个尚未排除的原因是它的 GT 全场统一
# （armature 0.1×12 / damping 4.5×12 / friction 0.05×12），且四条腿结构完全相同
# → 损失对腿间参数置换有精确对称性 → 实际有效维度远小于 49。
# S800 现在是 109 个逐关节不同的值，没有这个对称性。
# ⚠️ 打开时必须同时把 s800_pace_env_cfg.py 的 UNIFORM_GT 设为 True，否则 GT 会
#    落在 bounds 外 —— armature 尤其：现有 14 组分段区间的交集是空集
#    （max(lo)=0.05 > min(hi)=0.02），统一值无处可放。
# 注意：S800 的 27 个关节动力学各不相同（不像 ANYmal 四条腿相同），所以置换对称
#       性比 ANYmal 弱 —— 这个对照能排除维度，但不等价于复刻 ANYmal 的实验。
parser.add_argument("--uniform_gt", action=argparse.BooleanOptionalAction, default=False,
                    help="对照组：armature/damping/friction 的 GT 改为全场统一值。必须与 cfg 的 UNIFORM_GT 同步。")
parser.add_argument("--uniform_armature", type=float, default=0.05, help="--uniform_gt 时的统一 armature [kg m^2]。")
parser.add_argument("--uniform_damping", type=float, default=0.5, help="--uniform_gt 时的统一 viscous friction [Nm s/rad]。")
parser.add_argument("--uniform_friction", type=float, default=0.05, help="--uniform_gt 时的统一 coulomb friction [Nm]。取 0.05 是为了保证任何关节都不会被锁死（最小的 kp*A = 1.80 Nm）。")
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

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root

def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    if args_cli.num_envs != 1:
        raise ValueError("Data collection requires --num_envs 1")
    env_cfg.seed = args_cli.seed
    if env_cfg.sim2real.robot_name != "s800_sim":
        raise ValueError("This collector contains S800-specific truth and trajectories")
    if args_cli.scale_factor is not None and (args_cli.trajectory_profile != "scaled_legacy" or args_cli.trajectory_plan):
        raise ValueError("scale_factor requires scaled_legacy without a trajectory_plan")
    mode = args_cli.delay_mode or ("command" if args_cli.group != "all" else "torque")
    configure_delay(env_cfg, mode)
    if args_cli.self_collisions is not None:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args_cli.self_collisions
    elif args_cli.trajectory_profile == "clearance":
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    elif args_cli.group != "all":
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if env_cfg.decimation != 1:
        raise ValueError("PACE recording requires decimation=1")
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

    # ── ROUND-TRIP GROUND TRUTH ────────────────────────────────────────────
    # 这一组值是「注入仿真的已知真值」，用来验证 fit 能否把它恢复出来。
    # 不追求物理正确，但要求：
    #   1) 分组内 uniform、组间拉开 → 既能被辨识，又能暴露关节间串扰/错配
    #      （若 27 个关节全设同一个值，串扰发生了也看不出来）
    #   2) 量级随关节尺寸缩放 → 否则小关节会被灌进远大于自身的虚构惯量
    #   3) 必须落在 s800_pace_env_cfg.py 的 armature_bounds / damping_bounds 内
    #
    # 4 组: 0.24(大关节) / 0.14(髋侧摆) / 0.05(中小关节) / 0.008(末端)
    armature = torch.tensor([
        0.24, 0.14, 0.05, 0.24, 0.05, 0.05,  # 左腿: HIP_PITCH, HIP_ROLL, HIP_YAW, KNEE, ANKLE_P, ANKLE_R
        0.24, 0.14, 0.05, 0.24, 0.05, 0.05,  # 右腿
        0.05,                                 # 腰: TORSO_YAW
        0.05, 0.05, 0.05, 0.05, 0.008, 0.008, 0.008,  # 左臂: SHOULDER_P/R/Y, ELBOW_P, ELBOW_Y, WRIST_P/R
        0.05, 0.05, 0.05, 0.05, 0.008, 0.008, 0.008,  # 右臂
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

    # coulomb friction — 必须按关节缩放，不能用全场统一值。
    # 教训：曾用 [1.5]*27，结果手腕(最大PD力矩 12.5*0.05=0.625Nm)和肘偏航
    # (10.04*0.15=1.506Nm) 被 1.5Nm 的静摩擦完全锁死，前 3 秒位置几乎不变(变化量 1e-5 rad)。
    # 摩擦须明显小于该关节能产生的最大 PD 力矩，否则关节根本不动、参数不可辨识。
    #   大关节(HIP_PITCH/ROLL/KNEE) PD力矩 ~150-320Nm → 1.0
    #   中小关节                       PD力矩  4-70Nm   → 0.3
    #   末端(ELBOW_YAW/WRIST)          PD力矩 0.6-1.5Nm → 0.05
    friction = torch.tensor([
        1.0, 1.0, 0.3, 1.0, 0.3, 0.3,  # 左腿: HIP_PITCH, HIP_ROLL, HIP_YAW, KNEE, ANKLE_P, ANKLE_R
        1.0, 1.0, 0.3, 1.0, 0.3, 0.3,  # 右腿
        0.3,                                 # 腰: TORSO_YAW
        0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05,  # 左臂
        0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05,  # 右臂
    ], device=env.unwrapped.device).unsqueeze(0)

    # ── 对照组覆盖：把上面三组分段 GT 压成全场统一值 ─────────────────────
    # 用 fill_ 覆盖而不是重建，这样上面那套分组值仍然留在代码里作参考，
    # 也保证两条路径的 shape/dtype/device 完全一致。
    if args_cli.uniform_gt:
        armature.fill_(args_cli.uniform_armature)
        damping.fill_(args_cli.uniform_damping)
        friction.fill_(args_cli.uniform_friction)
        print(f"[uniform_gt] 对照组已启用: armature={args_cli.uniform_armature} "
              f"damping={args_cli.uniform_damping} friction={args_cli.uniform_friction}  (全场 27 关节)")
        print("[uniform_gt] ⚠️ 确认 s800_pace_env_cfg.py 的 UNIFORM_GT 已设为 True，否则真值落在 bounds 外。")

    bias = torch.tensor([0.05] * 27, device=env.unwrapped.device).unsqueeze(0)

    time_lag = torch.tensor([args_cli.delay_steps], dtype=torch.int, device=env.unwrapped.device)
    _, selected = select_groups(args_cli.group, joint_order)
    gt_vector = torch.cat([x.flatten() for x in (armature, damping, friction, bias, time_lag)])
    bounds = env_cfg.sim2real.bounds_params.to(gt_vector.device)
    if torch.any(gt_vector < bounds[:, 0]) or torch.any(gt_vector > bounds[:, 1]):
        raise ValueError("Injected truth is outside the fitting bounds")
    if not 0 <= args_cli.delay_steps <= min(a.max_delay for a in env_cfg.scene.robot.actuators.values()):
        raise ValueError("delay_steps outside actuator capacity")
    print(f"[experiment] group={args_cli.group}, joints={len(selected)}, delay={mode}/{args_cli.delay_steps} steps")

    env.reset()

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature), device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping), device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_viscous_friction_coeff[:, joint_ids] = damping
    # note: modeling coulomb friction if joint_friction = joint_dynamic_friction
    # If we set static friction lower than dynamic friction, the sim complains. So we need to do this weird order.
    articulation.write_joint_dynamic_friction_coefficient_to_sim(torch.zeros_like(friction), joint_ids=joint_ids, env_ids=torch.tensor([0], device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_dynamic_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0], device=env.unwrapped.device, dtype=torch.int32))
    articulation.data.joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids)
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
    t = torch.arange(num_steps, device=env.unwrapped.device) / sample_rate
    f0 = args_cli.min_frequency  # Hz
    f1 = args_cli.max_frequency  # Hz

    # ── 分组扫频 ───────────────────────────────────────────────────────────
    # 每个关节用自己的扫频上限。依据：各关节闭环带宽 ω_n = sqrt(kp/I_total) 差异很大，
    # 慢关节扫到高频只会"冻住"（实测 8Hz 段 HIP_PITCH 跟随度仅 5%），既浪费采集时间
    # 又在辨识里贡献无效样本；快关节则需要更高频率才能激励出 armature。
    #
    # 取值 ≈ 1.5 × ω_n（当前 I_total 校正增益下的实测值）:
    #   慢组 ω_n ≈ 2.0-2.4 Hz (HIP_PITCH/ROLL, KNEE, ANKLE_PITCH, TORSO,
    #                            SHOULDER_P/R, ELBOW_PITCH, WRIST_PITCH) → f1 = 3.0 Hz
    #   快组 ω_n ≈ 4.6-5.0 Hz (HIP_YAW, ANKLE_ROLL, SHOULDER_YAW,
    #                            ELBOW_YAW, WRIST_ROLL)                 → f1 = 6.0 Hz
    #
    # 附带好处：各关节相位不再相同（phase 含 (f1_j - f0) 项），激励自动去相关，
    # 回归矩阵条件数比全场同相位好很多。
    # 顺序 = joint_order（"J00_HIP_PITCH_L" ... "J33_WRIST_ROLL_R"）
    f1_per_joint = torch.tensor([
        3.0, 3.0, 6.0, 3.0, 3.0, 6.0,      # 左腿: HIP_P, HIP_R, HIP_Y, KNEE, ANK_P, ANK_R
        3.0, 3.0, 6.0, 3.0, 3.0, 6.0,      # 右腿
        3.0,                                # 腰
        3.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0, # 左臂: SH_P, SH_R, SH_Y, ELB_P, ELB_Y, WR_P, WR_R
        3.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0, # 右臂
    ], device=env.unwrapped.device)

    if not args_cli.grouped_sweep:
        f1_per_joint[:] = f1   # 关闭分组时退回全场统一 f1

    from s800_motion import nominal_motion, scaled_nominal_motion, choose_leg_motion
    from s800_clearance import LegClearance, with_midpoints
    center, amplitude = nominal_motion(joint_order)
    scale_factor = None
    if args_cli.trajectory_profile == "scaled_legacy" and not args_cli.trajectory_plan:
        scale_factor = args_cli.scale_factor if args_cli.scale_factor is not None else (.8 if args_cli.group in ("legs", "all") else 1.)
        center, amplitude = scaled_nominal_motion(joint_order, selected, scale_factor)
        print(f"[trajectory] original normalized bias/direction, scale factor={scale_factor}; no collision-free guarantee", flush=True)
    # Compute the normalized waveform on the actual simulation device first.
    t, wave = group_chirp(1 / sample_rate, duration, args_cli.rest, args_cli.ramp,
                         f0, f1_per_joint, torch.zeros(len(joint_order), device=env.unwrapped.device),
                         torch.ones(len(joint_order), device=env.unwrapped.device), selected)
    checker = None
    planned_report = None
    if args_cli.trajectory_profile == "clearance":
        checker = LegClearance(project_root() / "assets_robot/engineai_S800/urdf/serial_s800.urdf",
                               joint_order, base_height=env_cfg.scene.robot.init_state.pos[2])
    if args_cli.trajectory_plan:
        plan = json.loads(args_cli.trajectory_plan.read_text())
        if plan["joint_order"] != list(joint_order):
            raise ValueError("Trajectory plan joint order differs from environment")
        center = torch.tensor(plan["center_rad"], dtype=torch.float32)
        amplitude = torch.tensor(plan["signed_amplitude_rad"], dtype=torch.float32)
        if center.shape != (len(joint_order),) or amplitude.shape != center.shape:
            raise ValueError("Invalid trajectory plan dimensions")
        if not torch.isfinite(center).all() or not torch.isfinite(amplitude).all():
            raise ValueError("Nonfinite trajectory plan")
    elif checker is not None:
        center, amplitude, planned_report = choose_leg_motion(
            checker, wave.cpu().numpy(), center.numpy(), amplitude.numpy(), bias[0].cpu().numpy(),
            selected, args_cli.clearance_margin)
    commands = center.to(wave.device) + wave * amplitude.to(wave.device)
    if checker is not None:
        # Check the final float32 commands too (including an explicit user plan).
        physical_targets = (commands + bias).cpu().numpy()
        planned_report = checker.check(with_midpoints(physical_targets), margin=args_cli.clearance_margin + .03)
        planned_report.update(planning_reserve_m=.03, midpoint_checks=True,
                              continuous_collision_certificate=False)
        if not planned_report["passed"]:
            raise ValueError(f"Command trajectory failed clearance checks: {planned_report}")
        print(f"[clearance] planned={planned_report}", flush=True)
    print(f"[trajectory] center(rad)={center.tolist()}", flush=True)
    print(f"[trajectory] signed amplitude(rad)={amplitude.tolist()}", flush=True)
    num_steps = len(t)
    trajectory = torch.zeros((num_steps, articulation.num_joints), device=env.unwrapped.device)
    trajectory[:, joint_ids] = commands
    metadata = actuator_metadata(env, env_cfg, joint_ids, mode)
    metadata.update(group=args_cli.group, rest=args_cli.rest, ramp=args_cli.ramp,
                    duration=duration, f0=f0, f1_per_joint=f1_per_joint.cpu(),
                    trajectory_profile=args_cli.trajectory_profile, center_rad=center.cpu(),
                    signed_amplitude_rad=amplitude.cpu(), planned_clearance=planned_report,
                    scale_factor=scale_factor)

    articulation.write_joint_position_to_sim(commands[0].unsqueeze(0) + bias, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(bias), joint_ids=joint_ids)

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
            dof_target_pos_buffer[counter, :] = commands[counter]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter/sample_rate} seconds")
            if counter >= num_steps:
                break

    if counter != num_steps:
        raise RuntimeError("Simulation stopped before completing the recording")
    # Include the final post-step state in geometry checks, even though position
    # identification uses pre-step samples. Geometry never uses encoder readings directly.
    actual_report = None
    if checker is not None:
        physical_positions = torch.cat([dof_pos_buffer + bias,
            articulation.data.joint_pos[:, joint_ids]], dim=0).cpu().numpy()
        actual_report = checker.check(with_midpoints(physical_positions), margin=args_cli.clearance_margin)
        actual_report["time_s"] = actual_report["frame"] / 2 / sample_rate
        actual_report["midpoint_checks"] = True
        metadata["actual_clearance"] = actual_report
        print(f"[clearance] actual={actual_report}", flush=True)
    rejected = actual_report is not None and not actual_report["passed"]
    # close the simulator
    env.close()

    from time import sleep
    sleep(1)  # wait a bit for everything to settle

    (data_dir).mkdir(parents=True, exist_ok=True)
    output = Path(args_cli.output) if args_cli.output else data_dir / recording_name(args_cli.group, mode)
    if not output.is_absolute():
        output = project_root() / "data" / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if rejected:
        output = output.with_name(output.stem + ".rejected.pt")
    if output.exists():
        archived = output.with_name(output.stem + ".previous_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + output.suffix)
        output.rename(archived)
        old_plan = output.with_suffix(".trajectory.json")
        if old_plan.exists():
            old_plan.rename(archived.with_suffix(".trajectory.json"))
        print(f"[archive] {archived}", flush=True)
    plan = {"joint_order": list(joint_order), "center_rad": center.tolist(),
            "signed_amplitude_rad": amplitude.tolist(), "profile": args_cli.trajectory_profile,
            "planned_clearance": planned_report, "actual_clearance": actual_report,
            "limitations": "Sampled URDF visual/collision box envelopes; cross-leg/ground/leg-limits only, not continuous or whole-body certification."}
    output.with_suffix(".trajectory.json").write_text(json.dumps(plan, indent=2))
    torch.save({
        "time": time_data.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
        "dof_vel": dof_vel_buffer.cpu(),
        "dof_torque": dof_torque_buffer.cpu(),
        # ── 注入的真值（按 joint_order 排序），随数据一起落盘 ────────────────
        # 用途：fit.py 的 --floor_test 需要它把 bounds 收缩到真值；plot_trajectory
        # 出表时也应当优先用它，而不是 plot_trajectory.py 里手写的那份 GT 常量。
        # 手工表一旦和注入值对不上，误差不会报错、只会静默显示成错值 —— 这个坑踩过：
        # 26_09_29 那批数据是用旧增益采的，而拟合/出表用的是新增益，表现为几个关节的
        # 摩擦差 30~100%。数据自带真值可以从根上避免"口径对不上"这类问题。
        # 这里存的是「覆盖之后」的值（含 --uniform_gt 的影响）。
        "gt": {
            "joint_order": list(joint_order),
            "armature": armature[0].detach().float().cpu(),
            "damping": damping[0].detach().float().cpu(),
            "friction": friction[0].detach().float().cpu(),
            "bias": bias[0].detach().float().cpu(),
            "delay": float(time_lag[0].float().cpu()),
        },
        "joint_names": list(joint_order),
        "excited": [joint_order[i] for i in selected],
        "experiment": metadata,
    }, output)
    print(f"[saved] {output}", flush=True)
    if not args_cli.plot:
        if rejected:
            raise SystemExit("Actual motion failed clearance checks; diagnostic saved as .rejected.pt, do not fit it.")
        return

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
    if rejected:
        raise SystemExit("Actual motion failed clearance checks; diagnostic saved as .rejected.pt, do not fit it.")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
