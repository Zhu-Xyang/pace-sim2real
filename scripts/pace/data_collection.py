# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""采集辨识数据：给仿真注入已知真值 + 扫频激励，把轨迹落盘给 fit.py。

参数在哪（改参数只看这三处）
----------------------------
  1. 注入真值 + 激励设计 ....... scripts/pace/robot_tables.py  → ROBOT_TABLES[robot_name]
        armature / damping / friction      注入仿真的真值
        directions / bias / scale          激励整形 q = (sin(phase) + bias) × dir × scale
        f1_per_joint (分组扫频上限) / bias_gt / delay_gt
        命令行按关节名展开看：python scripts/pace/robot_tables.py --robot g1
  2. bounds / 关节顺序 / 增益 ... 各机器人的 env cfg
        source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/{s800,g1}_pace_env_cfg.py
  3. 扫频带 / 时长 .............. 本文件 CLI：--min_frequency --max_frequency --duration
        实际用的频带会写进 chirp_data.pt 的 "chirp" 字段（fit 的频段诊断据此对齐）

⚠️ 真值必须落在 cfg 的 bounds_params 内：越界时 CMA-ES 贴着边界收敛**且不报错**，
   表现是一个看起来正常的错值。启动时会自动检查（贴边 <5% 也告警）。

用法
----
    python scripts/pace/data_collection.py --task Isaac-Pace-G1-v0 --headless
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pace 数据采集：注入真值 + 扫频激励。")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Name of the task.")
# 频带（见文件顶部第 3 条）
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=None, help="Maximum frequency for the chirp signal in Hz。默认 None = 用 robot_tables 里该激励组的 f1（--excite all 时退回 10.0）。实际频带会写进 chirp_data.pt 的 'chirp' 字段，fit 的频段诊断据此自动对齐。")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
# 分组激励：只激励一部分关节，其余停在各自的扫描中心（见 robot_tables 的 "excite"）
parser.add_argument("--excite", type=str, default="all",
                    help="激励哪些关节: all（默认，全场扫频）| legs | upper | torso | arms。"
                         "非激励关节停在各自扫描中心（bias×dir×scale），不参与损失。"
                         "幅度与扫频上限按 robot_tables[robot]['excite'][组] 覆盖 —— "
                         "⚠️ 频带要和增益配套（极点要落在带内），改之前先看那张表的注释。")
parser.add_argument("--grouped_sweep", action=argparse.BooleanOptionalAction, default=False, help="EXPERIMENTAL, off by default. 每个关节用 robot_tables 里的 f1_per_joint 当扫频上限，外加逐关节相位偏移。实测比统一扫频**更差**：快关节跟不上、左右腿反相后互撞。要开就必须让左右对称关节同相位。")
# 对照组：全场统一真值（见 robot_tables.py 的 UNIFORM_CONTROL）
parser.add_argument("--uniform_gt", action=argparse.BooleanOptionalAction, default=False,
                    help="对照组：armature/damping/friction 的真值改成全场统一值。⚠️ 必须同时打开 cfg 的 UNIFORM_GT，否则统一值落在 bounds 外。")
parser.add_argument("--uniform_armature", type=float, default=0.05, help="--uniform_gt 时的统一 armature [kg m^2]。")
parser.add_argument("--uniform_damping", type=float, default=0.5, help="--uniform_gt 时的统一 viscous friction [Nm s/rad]。")
parser.add_argument("--uniform_friction", type=float, default=0.05, help="--uniform_gt 时的统一 coulomb friction [Nm]。取 0.05 是为了保证任何关节都不会被锁死（最小的 kp*A = 1.80 Nm）。")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root

# 注入真值 / 激励设计的单一数据源（纯字面量模块，见 robot_tables.py 顶部说明）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_tables import ROBOT_TABLES, group_indices, joint_type  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 1. 参数表 → 张量（+ 越界校验）
# ══════════════════════════════════════════════════════════════════════════
def load_gt_tensors(env_cfg, args, device):
    """从 ROBOT_TABLES 取该机器人的真值，应用 --uniform_gt 覆盖，返回按 joint_order 排列的张量。

    返回 dict: armature / damping / friction / encoder_bias / delay（+ robot / table）
    """
    robot = env_cfg.sim2real.robot_name
    if robot not in ROBOT_TABLES:
        raise SystemExit(f"[data_collection] ROBOT_TABLES 里没有 {robot}，先补一张表再跑。")
    T = ROBOT_TABLES[robot]
    joint_order = env_cfg.sim2real.joint_order
    n_joints = len(joint_order)

    # 长度不匹配会在 write_joint_*_to_sim 处才崩，这里提前拦下
    for key in ("armature", "damping", "friction", "f1_per_joint", "directions", "bias", "scale"):
        if len(T[key]) != n_joints:
            raise SystemExit(
                f"[data_collection] {robot} 的 {key} 长度 {len(T[key])} != joint_order 长度 {n_joints}。")

    armature = torch.tensor(T["armature"], dtype=torch.float32, device=device).unsqueeze(0)
    damping = torch.tensor(T["damping"], dtype=torch.float32, device=device).unsqueeze(0)
    friction = torch.tensor(T["friction"], dtype=torch.float32, device=device).unsqueeze(0)

    # ── 对照组覆盖：把三段真值压成全场统一值 ──────────────────────────────
    # 用 fill_ 覆盖而不是重建，保证两条路径的 shape/dtype 完全一致。
    if args.uniform_gt:
        # 统一值必须落在 bounds 内，而「全场统一 bounds」是 cfg 的 UNIFORM_GT 开关负责的。
        # 这里直接查 bounds 是否真的统一 —— 忘了同步 cfg 的话真值会落在区间外。
        _ab = env_cfg.sim2real.bounds_params[:n_joints]
        if not (torch.allclose(_ab[:, 0], _ab[0, 0]) and torch.allclose(_ab[:, 1], _ab[0, 1])):
            raise SystemExit(
                f"[uniform_gt] ❌ {robot} 的 cfg 里 armature bounds 不是全场统一的"
                f"（当前 [{_ab[:, 0].min():.4g}, {_ab[:, 1].max():.4g}]）。"
                f"请先把 env cfg 的 UNIFORM_GT 打开，否则真值越界、不会报错。")
        armature.fill_(args.uniform_armature)
        damping.fill_(args.uniform_damping)
        friction.fill_(args.uniform_friction)
        print(f"[uniform_gt] 对照组已启用: armature={args.uniform_armature} "
              f"damping={args.uniform_damping} friction={args.uniform_friction}  (全场 {n_joints} 关节)")

    gt = {
        "robot": robot,
        "table": T,
        "armature": armature,
        "damping": damping,
        "friction": friction,
        "encoder_bias": torch.full((1, n_joints), float(T["bias_gt"]), device=device),
        "delay": torch.tensor([[int(T["delay_gt"])]], dtype=torch.int, device=device),
    }
    check_gt_within_bounds(gt, env_cfg)
    print(f"[data_collection] robot={robot} ({n_joints} 关节)  参数来源: "
          f"scripts/pace/robot_tables.py ROBOT_TABLES['{robot}']"
          f"{'  + --uniform_gt 覆盖' if args.uniform_gt else ''}")
    print(f"    armature {armature.min():.4g}~{armature.max():.4g}   "
          f"damping {damping.min():.4g}~{damping.max():.4g}   "
          f"friction {friction.min():.4g}~{friction.max():.4g}   "
          f"bias_gt {T['bias_gt']}  delay_gt {T['delay_gt']}")
    return gt


def check_gt_within_bounds(gt, env_cfg):
    """校验**注入用的**真值落在 cfg 的 bounds_params 内。

    ⚠️ 必须查覆盖之后的张量（--uniform_gt 会改），查原表会在对照组下误判。
    越界时 CMA-ES 贴着边界收敛且不报错，而且 bounds 中点（= 搜索初值）会离真值很远。
    （手腕摩擦踩过这个坑：XML 给 0.1，原 bounds 上界只有 1%×5Nm = 0.05。）
    """
    order = env_cfg.sim2real.joint_order
    n = len(order)
    b = env_cfg.sim2real.bounds_params.cpu()
    for sl, name, value in ((slice(0, n), "armature", gt["armature"]),
                            (slice(n, 2 * n), "damping", gt["damping"]),
                            (slice(2 * n, 3 * n), "friction", gt["friction"])):
        lo, hi = b[sl, 0], b[sl, 1]
        g = value[0].detach().cpu().float()
        out = (g < lo - 1e-9) | (g > hi + 1e-9)
        if out.any():
            bad = [(order[i], float(g[i]), float(lo[i]), float(hi[i])) for i in out.nonzero().flatten().tolist()]
            raise SystemExit(
                f"[data_collection] ❌ 注入真值越界（{name}）：{bad[:6]}{' ...' if len(bad) > 6 else ''}\n"
                f"     越界时 CMA-ES 会贴着边界收敛且不报错。请先放宽 cfg 的 bounds。")
        # 贴边告警：真值落在区间外侧 5% 之内时，搜到边界就停不下来
        margin = torch.minimum(g - lo, hi - g) / (hi - lo).clamp_min(1e-12)
        tight = (margin < 0.05).nonzero().flatten().tolist()
        if tight:
            print(f"[data_collection] ⚠️ {name} 有 {len(tight)} 个关节的真值贴近区间边界"
                  f"（余量 <5%）: {[order[i] for i in tight[:6]]}")

    # delay 也要查：它是全局标量，不在上面那三段里。越界同样只会静默贴边；
    # ⚠️ 而且它还有第二道约束 —— DelayBuffer 只允许 0..max_delay+1，采样越界会直接
    #    ValueError（不是静默错），所以 bounds 上界必须 ≤ max_delay+1。
    _d_lo, _d_hi = float(b[4 * n, 0]), float(b[4 * n, 1])
    _d_gt = float(gt["delay"][0, 0])
    if not (_d_lo - 1e-9 <= _d_gt <= _d_hi + 1e-9):
        raise SystemExit(
            f"[data_collection] ❌ 注入的 delay 真值 {_d_gt} 越界（bounds [{_d_lo}, {_d_hi}]）。"
            f"越界时 CMA-ES 会贴着边界收敛且不报错。")
    _max_delay = getattr(env_cfg.scene.robot.actuators.get("joints"), "max_delay", None)
    if _max_delay is not None and _d_hi > _max_delay + 1:
        raise SystemExit(
            f"[data_collection] ❌ delay bounds 上界 {_d_hi} > max_delay+1 = {_max_delay + 1}"
            f"（DelayBuffer 允许的最大滞后）。采样越界会 ValueError，请调 cfg 的 max_delay。")
    print(f"[data_collection] ✓ delay 真值 {_d_gt:g} 步在 bounds [{_d_lo:g}, {_d_hi:g}] 内"
          f"（max_delay={_max_delay}）")


def apply_gt_to_sim(articulation, gt, joint_ids):
    """把真值写进仿真：armature / viscous friction / coulomb friction / 延时 / 编码器零位。

    ⚠️ 只写 env 0（采集只用单环境；多环境时其余环境的真值没被覆盖，要一并改这里）。
    """
    dev = joint_ids.device
    one = torch.tensor([0], device=dev, dtype=torch.int32)          # 只写 env 0
    all_envs = torch.arange(articulation.num_instances, device=dev, dtype=torch.int32)

    articulation.write_joint_armature_to_sim(gt["armature"], joint_ids=joint_ids, env_ids=one)
    articulation.data.joint_armature[:, joint_ids] = gt["armature"]
    articulation.write_joint_viscous_friction_coefficient_to_sim(gt["damping"], joint_ids=joint_ids, env_ids=one)
    articulation.data.joint_viscous_friction_coeff[:, joint_ids] = gt["damping"]
    # note: modeling coulomb friction if joint_friction = joint_dynamic_friction
    # If we set static friction lower than dynamic friction, the sim complains. So we need to do this weird order.
    articulation.write_joint_dynamic_friction_coefficient_to_sim(gt["friction"], joint_ids=joint_ids, env_ids=one)
    articulation.data.joint_dynamic_friction_coeff[:, joint_ids] = gt["friction"]
    articulation.write_joint_friction_coefficient_to_sim(gt["friction"], joint_ids=joint_ids, env_ids=one)
    articulation.data.joint_friction_coeff[:, joint_ids] = gt["friction"]

    # 延时 / 编码器零位：驱动器按关节名分组，要把「配置顺序」映射成各驱动器的列序
    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=dev)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(gt["delay"])
        articulation.actuators[drive_type].update_encoder_bias(gt["encoder_bias"][:, drive_joint_idx])
        articulation.actuators[drive_type].reset(all_envs)


# ══════════════════════════════════════════════════════════════════════════
# 2. 激励轨迹
# ══════════════════════════════════════════════════════════════════════════
def build_trajectory(env, T, joint_order, joint_ids, args):
    """线性 chirp + 逐关节整形 → (trajectory, t, f1_per_joint, duration, sample_rate, excited_idx)。

    trajectory 的列是**仿真关节顺序**（要直接喂 action）；joint_ids[j] = 配置关节 j 的仿真列号。
    """
    dev = env.unwrapped.device
    duration = args.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=dev)
    f0 = args.min_frequency

    # ── 分组激励：只扫一部分关节 ─────────────────────────────────────────
    # 被选中的关节用 robot_tables[robot]["excite"][组] 的幅度/频带覆盖；**其余关节停在
    # 各自的扫描中心**（bias×dir×scale = 原激励的零点姿态），而不是停在 q=0 ——
    # G1 的手臂在零位贴身体，停在 0 会自撞（自撞会让支链通过接触力耦合，破坏
    # 分组优化的解耦前提）。
    excited_idx = group_indices(joint_order, args.excite)
    scale_list = list(T["scale"])
    band = args.max_frequency
    if args.excite != "all":
        exc = T.get("excite", {}).get(args.excite)
        if exc is None:
            raise SystemExit(f"[excite] {T.get('robot', '该机器人')} 的 robot_tables 里没有 excite['{args.excite}']，"
                             f"先补上（含 f1 与逐类型 scale）再跑。")
        for j, name in enumerate(joint_order):
            t_name = joint_type(name)
            if t_name in exc.get("scale", {}):
                scale_list[j] = exc["scale"][t_name]
        if band is None:
            band = exc.get("f1", 10.0)
        print(f"[excite] 只激励 {args.excite}: {len(excited_idx)} 个关节"
              f"（{', '.join(joint_type(joint_order[i]) for i in excited_idx)}）；"
              f"其余 {len(joint_order) - len(excited_idx)} 个停在扫描中心；扫频 0.1→{band:g}Hz")
    if band is None:
        band = 10.0
    f1 = band

    # 分组扫频：每个关节用自己的扫频上限（各关节闭环带宽 ω_n 差异很大，慢关节扫到高频
    # 只会"冻住"——实测 8Hz 段 HIP_PITCH 跟随度仅 5%，既浪费采集又贡献无效样本）。
    # ⚠️ 每台机器人的 ω_n 不同，f1_per_joint 存在 robot_tables.py，换机器人要按它的 ω_cl 重设。
    f1_per_joint = torch.tensor(T["f1_per_joint"], device=dev, dtype=torch.float32)
    if not args.grouped_sweep:
        f1_per_joint[:] = f1   # 关闭分组时退回全场统一 f1

    # 逐关节独立相位 —— 仅用于诊断激励共线，默认关闭。
    # ⚠️ 实测代价：左右腿失去同步，HIP_ROLL 左右反向摆动导致**双腿互撞**；且跟踪变差。
    # cond(X^T X) 只是「指令矩阵」的条件数，是可辨识性的代理指标而非判据 —— fit 匹配的是
    # 仿真轨迹，参数→轨迹经过非线性刚体动力学，指令共线并不等于参数不可辨识。
    if args.grouped_sweep:
        phase_offset = 2 * pi * torch.arange(len(joint_ids), device=dev) / len(joint_ids)
    else:
        phase_offset = torch.zeros(len(joint_ids), device=dev)

    # 线性 chirp: phase_k(t) = 2*pi*(f0*t + (f1_k-f0)/(2*duration)*t^2) + phi_k
    phase = (2 * pi * (f0 * t[:, None] + ((f1_per_joint[None, :] - f0) / (2 * duration)) * t[:, None] ** 2)
             + phase_offset[None, :])
    chirp_signal = torch.sin(phase)                       # (num_steps, n_joints)，按 joint_order 排列

    # 索引方向：trajectory 的列是「仿真关节顺序」，而 joint_order/bias/scale 是「配置顺序」。
    # joint_ids[j] = 配置关节 j 对应的仿真列号，故用 trajectory[:, joint_ids] = X
    # 把 X 的第 j 列写到仿真列 joint_ids[j]。
    trajectory = torch.zeros((num_steps, len(joint_ids)), device=dev)
    trajectory[:, joint_ids] = chirp_signal
    # 整形：q = (sin(phase) + bias) × direction × scale，三个数组的取值与理由见 robot_tables.py
    trajectory_directions = torch.tensor(T["directions"], device=dev)
    trajectory_bias = torch.tensor(T["bias"], device=dev)
    trajectory_scale = torch.tensor(scale_list, device=dev)
    center = trajectory_bias * trajectory_directions * trajectory_scale      # 各关节的扫描中心
    trajectory[:, joint_ids] = ((trajectory[:, joint_ids] + trajectory_bias.unsqueeze(0))
                                * trajectory_directions.unsqueeze(0) * trajectory_scale.unsqueeze(0))
    # 非激励关节：整条轨迹冻结在扫描中心
    if len(excited_idx) < len(joint_order):
        _exc = set(excited_idx)
        _frozen = torch.tensor([j for j in range(len(joint_order)) if j not in _exc],
                               dtype=torch.long, device=dev)
        trajectory[:, joint_ids[_frozen]] = center[_frozen].unsqueeze(0)
    return trajectory, t, f1_per_joint, duration, sample_rate, excited_idx


# ══════════════════════════════════════════════════════════════════════════
# 3. rollout
# ══════════════════════════════════════════════════════════════════════════
def rollout(env, trajectory, joint_ids, encoder_bias, sample_rate):
    """把 trajectory 逐步喂成 action，记录 pos / target / vel / torque。

    所有 buffer 都是**配置顺序**（joint_order），与落盘的 gt 对齐。
    """
    dev = env.unwrapped.device
    num_steps, n = trajectory.shape
    dof_pos_buffer = torch.zeros(num_steps, n, device=dev)
    dof_target_pos_buffer = torch.zeros(num_steps, n, device=dev)
    dof_vel_buffer = torch.zeros(num_steps, n, device=dev)
    dof_torque_buffer = torch.zeros(num_steps, n, device=dev)

    # 从轨迹首帧起步，避免第一拍有跳变
    articulation = env.unwrapped.scene["robot"]
    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + encoder_bias[0, joint_ids])
    articulation.write_joint_velocity_to_sim(torch.zeros((1, len(joint_ids)), device=dev))

    counter = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            robot = env.unwrapped.scene.articulations["robot"]
            dof_pos_buffer[counter, :] = robot.data.joint_pos[0, joint_ids] - encoder_bias[0]
            dof_vel_buffer[counter, :] = robot.data.joint_vel[0, joint_ids]
            dof_torque_buffer[counter, :] = robot.data.applied_torque[0, joint_ids]
            actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            obs, _, _, _, _ = env.step(actions)
            dof_target_pos_buffer[counter, :] = robot._data.joint_pos_target[0, joint_ids]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter / sample_rate} seconds")
            if counter >= num_steps:
                break
    return dof_pos_buffer, dof_target_pos_buffer, dof_vel_buffer, dof_torque_buffer


# ══════════════════════════════════════════════════════════════════════════
# 4. 落盘
# ══════════════════════════════════════════════════════════════════════════
def dataset_path(data_dir, excite):
    """数据文件名。--excite all 就是 chirp_data.pt；分组激励另存一个名字，
    **不要覆盖全身数据**（那是另一轮实验的输入，共享路径被覆盖后 fit 的 config.pt
    是唯一快照，出表/复核都只能靠它）。"""
    return data_dir / ("chirp_data.pt" if excite == "all" else f"chirp_data_{excite}.pt")


def save_dataset(data_dir, t, buffers, gt, args, duration, f1_per_joint, joint_order, excited_idx):
    """存 chirp_data*.pt。下游（fit / plot_trajectory）按文件名取。"""
    dof_pos_buffer, dof_target_pos_buffer, dof_vel_buffer, dof_torque_buffer = buffers
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "time": t.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
        "dof_vel": dof_vel_buffer.cpu(),
        "dof_torque": dof_torque_buffer.cpu(),
        # ── 注入的真值（按 joint_order 排序），随数据一起落盘 ────────────────
        # 目的：下游（plot_trajectory 出表、事后复核）不用再手工维护一张 GT 表。
        # 手工表一旦和注入值对不上，误差不会报错、只会静默显示成 0% —— 表现是
        # 「所有关节都完美拟合」，这个坑踩过两次（uniform 那轮按分段 GT 打分、
        # 以及新机器人名字查不到）。这里存的是「覆盖之后」的值，所以 --uniform_gt
        # 的口径会被自动记录下来。
        "gt": {
            "joint_order": list(gt["joint_order"]),
            "armature": gt["armature"][0].detach().float().cpu(),
            "damping": gt["damping"][0].detach().float().cpu(),
            "friction": gt["friction"][0].detach().float().cpu(),
            "bias": gt["encoder_bias"][0].detach().float().cpu(),
            "delay": float(gt["delay"][0, 0].float().cpu()),
        },
        # ── 实际用的扫频带，随数据一起落盘 ──────────────────────────────────
        # fit 的「段号 → 频段」映射以前靠配置里的手写常量对齐，错位过一轮
        # （数据 0.1-2 / 0.1-4 / 0.1-10Hz 三种，配置恒为 0.1-4）—— 不报错，只是 TB 里
        # 5_BandLoss/* 的标签全错（损失本身不受影响，它一直覆盖全部时间步）。
        "chirp": {
            "kind": "grouped" if args.grouped_sweep else "linear",
            "f0": float(args.min_frequency),
            "f1_per_joint": [float(x) for x in f1_per_joint],
            "duration": float(duration),
        },
        # ── 哪些关节被激励（--excite）────────────────────────────────────────
        # 非激励关节的轨迹是常数（冻结在扫描中心），不含激励信息、不该进拟合。
        # fit.py 会拿它与 --group 对一下，错配时告警。
        "excited": [joint_order[i] for i in excited_idx],
    }, dataset_path(data_dir, args.excite))


# ══════════════════════════════════════════════════════════════════════════
# 5. 采集后体检图
# ══════════════════════════════════════════════════════════════════════════
def plot_diagnostics(t, buffers, gt, joint_order, title_prefix):
    """位置 / 力矩 / 速度三张图。看关节有没有被锁死、贴限位、自激。"""
    import matplotlib.pyplot as plt

    dof_pos_buffer, dof_target_pos_buffer, dof_vel_buffer, dof_torque_buffer = buffers
    n_joints = len(joint_order)
    n_cols = 4
    n_rows = (n_joints + n_cols - 1) // n_cols
    t_np = t.cpu().numpy()

    # friction_torque = viscous * q̇ + coulomb * sign(q̇)；net_torque = applied - friction
    viscous = gt["damping"][0].cpu()
    coulomb = gt["friction"][0].cpu()
    vel_cpu = dof_vel_buffer.cpu()
    torque_cpu = dof_torque_buffer.cpu()
    friction_torque = viscous.unsqueeze(0) * vel_cpu + coulomb.unsqueeze(0) * torch.sign(vel_cpu)
    net_torque = torque_cpu - friction_torque

    def _grid():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True)
        return fig, (axes.flatten() if n_joints > 1 else [axes])

    # ── Figure 1: joint position ──
    fig, axes = _grid()
    for i in range(n_joints):
        ax = axes[i]
        ax.plot(t_np, dof_pos_buffer[:, i].cpu().numpy(), label="pos")
        ax.plot(t_np, dof_target_pos_buffer[:, i].cpu().numpy(), label="target", linestyle='dashed')
        ax.set_title(joint_order[i], fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Pos [rad]", fontsize=8)
        ax.grid(True)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    for i in range(n_joints, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f"{title_prefix} — Joint Position (chirp tracking)", fontsize=12)
    plt.tight_layout()

    # ── Figure 2: applied torque & net torque (after friction) ──
    fig2, axes2 = _grid()
    for i in range(n_joints):
        ax = axes2[i]
        ax.plot(t_np, torque_cpu[:, i].numpy(), label="applied_torque", linewidth=0.8)
        ax.plot(t_np, net_torque[:, i].numpy(), label="net_torque", linewidth=0.8, linestyle='dashed')
        ax.set_title(joint_order[i], fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Torque [Nm]", fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
    for i in range(n_joints, len(axes2)):
        axes2[i].set_visible(False)
    fig2.suptitle(f"{title_prefix} — Applied Torque vs Net Torque (after friction)", fontsize=12)
    plt.tight_layout()

    # ── Figure 3: joint velocity ──
    fig3, axes3 = _grid()
    for i in range(n_joints):
        ax = axes3[i]
        ax.plot(t_np, vel_cpu[:, i].numpy(), label="vel", color="tab:blue", linewidth=0.8)
        ax.set_title(joint_order[i], fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Vel [rad/s]", fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for i in range(n_joints, len(axes3)):
        axes3[i].set_visible(False)
    fig3.suptitle(f"{title_prefix} — Joint Velocity", fontsize=12)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order],
                             device=env.unwrapped.device, dtype=torch.int32)
    print("Isaac Lab joint names:", articulation.joint_names)
    print("Config joint_order:", joint_order)
    print("Mapped joint_ids:", joint_ids)

    # ── 1. 真值 ──
    gt = load_gt_tensors(env_cfg, args_cli, env.unwrapped.device)
    gt["joint_order"] = joint_order

    # ── 2. 激励 ──
    T = gt["table"]
    trajectory, t, f1_per_joint, duration, sample_rate, excited_idx = build_trajectory(
        env, T, joint_order, joint_ids, args_cli)

    # ── 3. 注入 + rollout ──
    env.reset()
    apply_gt_to_sim(articulation, gt, joint_ids)
    buffers = rollout(env, trajectory, joint_ids, gt["encoder_bias"], sample_rate)
    env.close()
    from time import sleep
    sleep(1)  # wait a bit for everything to settle

    # ── 4. 落盘 + 体检图 ──
    data_dir = project_root() / "data" / gt["robot"]
    save_dataset(data_dir, t, buffers, gt, args_cli, duration, f1_per_joint, joint_order, excited_idx)
    print(f"[data_collection] ✓ 已保存 {dataset_path(data_dir, args_cli.excite)}")
    plot_diagnostics(t, buffers, gt, joint_order, f"{gt['robot']} Chirp")


if __name__ == "__main__":
    main()
    simulation_app.close()
