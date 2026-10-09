# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
from experiment_utils import (build_joint_groups, select_groups, recording_name, configure_delay,
                              actuator_metadata, validate_recording, freeze_unfitted_truth)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Name of the task.")

# Only selected joints enter the optimization and loss. Other joints remain simulated.
# Torso and arms are coupled: use a measured prior/warm start, or explicitly request
# --freeze_unfitted_gt for a controlled sim-to-sim recovery experiment.
parser.add_argument("--data", type=str, help="数据路径；相对路径从 data/ 起算。")
parser.add_argument("--delay_mode", choices=("command", "torque"), default=None,
                    help="默认跟随录制元数据；分组新实验默认 command。")
parser.add_argument("--fix_delay", type=int, default=None, help="固定整数延时，不参与搜索。")
parser.add_argument("--freeze_unfitted_gt", action="store_true",
                    help="仅 sim-to-sim：未拟合组用记录真值冻结；本组仍从先验或 warm start 搜索。")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=Path, default=None, help="日志根目录；每次运行仍创建独立子目录。")
parser.add_argument("--group", type=str, default="all",
                    help="要优化的关节组，逗号分隔: all | legs | arms | torso（可组合，如 legs,torso）。")
parser.add_argument("--warm_start", type=str, default=None,
                    help="mean_*.pt 路径，或包含 mean_*.pt 的 run 目录（自动取迭代数最大的一个）。"
                         "未参与优化的组取这里的值，同时作为 CMA-ES 的初始 mean。")
parser.add_argument("--opt_delay", action=argparse.BooleanOptionalAction, default=True,
                    help="每轮是否把全局 delay 参数一起放开重新优化（默认放开；关掉则冻结在 warm_start 上）。")
parser.add_argument("--sigma", type=float, default=None,
                    help="覆盖 cfg 里的 CMA-ES 初始 sigma。warm_start 是「接着细化」时通常要调小。")
parser.add_argument("--epsilon", type=float, default=None,
                    help="覆盖 cfg 的提前停止判据（种群 (max-min)/min < epsilon 即判收敛）。"
                         "分组实验默认设 0 关闭提前停止，按指定迭代次数运行。")
parser.add_argument("--floor_test", action="store_true",
                    help="一致性检查：把 bounds 收缩到数据文件里的注入真值 ±1e-6，只跑 1 代。"
                         "若采集与拟合在时间对齐/零偏/延时/初始状态上完全一致，真值参数应当能"
                         "逐位复现数据（损失 <1e-10）。**这是识别流程最重要的一步自检** —— "
                         "不通过就说明数据与模型口径不一致，任何拟合结果都不可信。"
                         "建议 --num_envs 32 即可（框已收缩，样本几乎相同）。")

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
from pace_sim2real import CMAESOptimizer


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None and args_cli.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    active_groups, selected = select_groups(args_cli.group, env_cfg.sim2real.joint_order)
    group_label = "all" if active_groups is None else "_".join(active_groups)
    default_mode = "command" if active_groups else "torque"
    requested_mode = args_cli.delay_mode or default_mode
    default_data = (Path(env_cfg.sim2real.data_dir) if active_groups is None else
                    Path(env_cfg.sim2real.robot_name) / recording_name(args_cli.group, requested_mode))
    data_file = Path(args_cli.data) if args_cli.data else default_data
    if not data_file.is_absolute():
        data_file = project_root() / "data" / data_file
    data = torch.load(data_file, map_location="cpu", weights_only=True)
    mode = args_cli.delay_mode or data.get("experiment", {}).get("delay_mode", default_mode)
    configure_delay(env_cfg, mode)
    if "self_collisions" in data.get("experiment", {}):
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = data["experiment"]["self_collisions"]
    if env_cfg.decimation != 1:
        raise ValueError("PACE replay requires decimation=1")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create optimization
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device, dtype=torch.int32)

    log_name = env_cfg.sim2real.robot_name if active_groups is None else f"{env_cfg.sim2real.robot_name}_{group_label}_{mode}"
    log_dir = args_cli.log_dir or project_root() / "logs" / "pace" / log_name
    validate_recording(data, joint_order, selected, actuator_metadata(env, env_cfg, sim_joint_ids, mode))
    print(f"[experiment] {data_file}, group={group_label}, delay={mode}, loss joints={len(selected)}")
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    # ── 一致性检查（floor test）：把 bounds 收缩到真值本身 ──────────────────
    # 采集和拟合若在时间对齐 / 零偏 / 延时 / 初始状态上完全一致，把真值原样放进去，
    # 损失应该只剩浮点噪声（<1e-10）。对不上就说明两边口径有系统性差异 ——
    # 这时任何拟合结果都不可信，先修流程再谈参数。
    # ⚠️ 真值取自**数据文件**（data_collection.py 随数据落盘的 "gt"），不是代码里的常量：
    #    代码常量与注入值漂移过一次（26_09_29 用旧增益采、新增益拟合），数据自带真值
    #    是唯一能避免这类"口径对不上"的做法。
    if args_cli.floor_test:
        if "gt" not in data:
            raise SystemExit(
                "[floor_test] 数据文件里没有 'gt' 字段（老数据）。请先用当前 data_collection.py "
                "重采一次，或在 data_collection.py 的 torch.save 里确认写了 \"gt\"。")
        _gt = data["gt"]
        if list(_gt["joint_order"]) != list(joint_order):
            raise SystemExit("[floor_test] 数据里的 joint_order 与 env cfg 不一致，先对齐再跑。")
        _gt_vec = torch.cat([_gt["armature"].flatten(), _gt["damping"].flatten(),
                             _gt["friction"].flatten(), _gt["bias"].flatten(),
                             torch.tensor([float(_gt["delay"])])]).cpu().to(bounds_params.dtype)
        _eps = 1e-6 * _gt_vec.abs().clamp_min(1e-6)
        bounds_params = torch.stack([_gt_vec - _eps, _gt_vec + _eps], dim=1).to(bounds_params.device)
        print(f"[floor_test] bounds 已收缩到数据里的注入真值 ±1e-6 相对，只跑 1 代"
              f"（真值 delay={float(_gt['delay']):g} 步）")

    # ── 分组优化：解析要优化的组 + 载入 warm start ────────────────────────
    groups = build_joint_groups(joint_order)
    print("[group] 分组: " + ", ".join(f"{g}={len(v)}关节" for g, v in groups.items()))

    if active_groups and not args_cli.freeze_unfitted_gt and not args_cli.floor_test:
        print("[group] 未拟合参数来自 warm_start 或 bounds 中点；torso/arms 的动力学耦合仍存在。")

    warm_start = None
    if args_cli.warm_start:
        ws_path = Path(args_cli.warm_start)
        if ws_path.is_dir():
            cands = sorted(ws_path.glob("mean_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
            if not cands:
                raise SystemExit(f"[group] {ws_path} 下没找到 mean_*.pt")
            ws_path = cands[-1]
        warm_start = torch.load(ws_path, map_location="cpu")
        print(f"[group] warm_start = {ws_path}（{warm_start.numel()} 个参数）")

    if args_cli.floor_test:
        # Floor checks must initialize every frozen dimension at truth too.
        warm_start = None
    elif args_cli.freeze_unfitted_gt:
        warm_start = freeze_unfitted_truth(bounds_params, data, joint_order, selected, warm_start)
        print("[diagnostic] 非本组参数冻结为注入真值；这是条件参数恢复实验，不代表真机未知全参数辨识。")

    if args_cli.fix_delay is not None:
        delay = args_cli.fix_delay
        if not bounds_params[-1, 0] <= delay <= bounds_params[-1, 1]:
            raise ValueError("fix_delay outside parameter bounds")
        if delay > min(a.max_delay for a in env_cfg.scene.robot.actuators.values()):
            raise ValueError("fix_delay exceeds actuator buffer")
        bounds_params[-1] = torch.tensor([delay - 1e-6, delay + 1e-6], device=bounds_params.device)
        if warm_start is not None:
            warm_start[-1] = delay
        args_cli.opt_delay = False

    # Rest/taper and heterogeneous chirps cannot use the old time-to-frequency labels.
    segment_edges = () if "experiment" in data else env_cfg.sim2real.cmaes.segment_edges_hz
    data["fit"] = {"groups": active_groups, "fitted_joints": selected,
                   "freeze_unfitted_gt": args_cli.freeze_unfitted_gt, "floor_test": args_cli.floor_test,
                   "data_file": str(data_file), "delay_mode": mode, "seed": args_cli.seed,
                   "fix_delay": args_cli.fix_delay, "warm_start": args_cli.warm_start}

    opt = CMAESOptimizer(
        bounds=bounds_params,
        groups=groups,
        active_groups=active_groups,
        warm_start=warm_start,
        opt_delay=args_cli.opt_delay,
        loss_joints=selected,
        seed=args_cli.seed,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=1 if args_cli.floor_test else (env_cfg.sim2real.cmaes.max_iteration if args_cli.max_iterations is None else args_cli.max_iterations),
        data=data,
        device=env.unwrapped.device,
        epsilon=0.0 if args_cli.floor_test else (
            (0.0 if active_groups else env_cfg.sim2real.cmaes.epsilon) if args_cli.epsilon is None else args_cli.epsilon),
        sigma=env_cfg.sim2real.cmaes.sigma if args_cli.sigma is None else args_cli.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval if args_cli.save_interval is None else args_cli.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
        segment_edges_hz=segment_edges,
        sweep_kind=env_cfg.sim2real.cmaes.sweep_kind,
    )

    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            opt.tell(env.unwrapped.scene.articulations["robot"].data.joint_pos[:, sim_joint_ids], measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1))
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)
    # close optimizer
    opt.close()
    # close the simulator
    env.close()

    if args_cli.floor_test:
        _s = opt.scores_buffer[0]                     # 第 1 代全种群的分数
        _best, _worst = _s.min().item(), _s.max().item()
        _n_j = len(selected)
        _rms = (_best / max(_n_j, 1)) ** 0.5          # 折成逐关节 RMS 残差 [rad]
        _at_floor = int((_s < 1e-10).sum())
        print("\n" + "=" * 72)
        print(f"[floor_test] 真值原样放进 {_s.numel()} 个环境，只跑 1 代：")
        print(f"    损失  min {_best:.3e}  max {_worst:.3e}   （{_n_j} 个关节的平方误差和，rad²）")
        print(f"    折算  逐关节 RMS 残差 {_rms:.3e} rad")
        print(f"    达到浮点地板(<1e-10) 的样本：{_at_floor}/{_s.numel()}")

        # ── 残差的时间结构：区分「口径不一致」与「数据里有非光滑事件」 ──────
        # 口径不一致 ⇒ 残差从第 0 步就存在、且随时间平稳；
        # 非光滑事件（限位冲击/自碰撞/自激）⇒ **开局几十步仍在浮点地板**，
        # 到某一刻被冲击炸开，之后再衰减。后者不是流程的错，是激励设计的错。
        _bi = int(torch.argmin(_s).item())
        _sim_best = opt.sim_dof_pos_buffer[_bi].cpu()                 # (T, n)
        _real = data["dof_pos"].cpu() + _gt["bias"].unsqueeze(0)      # 真实位置 = 记录值 + 零位
        _res = (_sim_best - _real)[:, selected].abs()
        _w = max(1, int(0.05 * _res.shape[0]))                        # 前 5% 步（约 1 s）
        _early = float(_res[:_w].pow(2).mean().sqrt())
        _late = float(_res[-_w:].pow(2).mean().sqrt())
        _per_step = _res.pow(2).mean(1).sqrt()
        # 注意 argmax 不支持 Bool 张量（torch 会报 "argmax_cpu not implemented for 'Bool'"），
        # 用 nonzero 取第一个 True 的下标。
        _mask = _per_step > max(50 * _early, 1e-6)
        _jump = int(torch.nonzero(_mask)[0].item()) if bool(_mask.any()) else -1
        print(f"    时间结构  开局 5% 时段 RMS {_early:.2e} rad   末段 {_late:.2e} rad")
        if _jump >= 0:
            print(f"              首次冲击（残差跳出 50×开局）出现在第 {_jump} 步 "
                  f"= {float(data['time'][_jump]):.2f} s")
        print("              开局吻合只验证初期时序；是否通过以完整轨迹损失为准。")

        # 判定看 min：口径一致时**存在**一组参数能逐位复现数据。用 max 判会被整数量化
        # 误伤 —— 框收缩到真值 ±1e-6 时，若真值压在延时格子边界上，一半样本会取到相邻的
        # 整数延时，损失差好几个数量级（那是取整方式的问题，不是口径不一致）。
        if _best < 1e-10:
            print("    ✓ PASS —— 采集与拟合口径一致（时间对齐/零偏/延时/初始状态）")
            if _at_floor < _s.numel():
                print(f"    （{_s.numel() - _at_floor} 个样本未达到阈值，仍需检查数值敏感性及接触/限位事件。）")
        else:
            print("    ✗ FAIL —— 全程回放未达到一致性阈值，请先检查以下原因：")
            print("      · 数据是用**别的 cfg** 采的（增益/URDF/sim_dt 改过之后没重采）")
            print("      · 初始状态或时间对齐差一步（des_dof_pos 与 dof_pos 错位）")
            print("      · 代码里的真值常量与注入值不一致（本检查读的是数据里的 gt，不是常量）")
            print("      · 接触/限位事件，或动力学对参数微扰的敏感性")
            raise SystemExit(1)
        print("=" * 72)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
