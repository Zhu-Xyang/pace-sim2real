# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Name of the task.")

# ── 分组优化（手 / 腿 / 腰）──────────────────────────────────────────────
# 每轮只优化一个组，其余组冻结在 --warm_start 给出的值上（不给就从 bounds 中点起步）。
#
# URDF 是三条挂在 LINK_BASE 上的独立支链（左腿 / 右腿 / 腰→双臂），基座焊死
# （fix_root_link=True）⇒ 质量阵支链间块对角、损失可加：腿的结果不受臂/腰冻结值影响。
# 但支链内耦合：torso 是双臂的父关节、臂对 torso 有反作用 ⇒ 这两个必须同一轮。
#
# 从零开始的顺序（不给 warm_start ⇒ 本组从中点全空间搜索）：
#   python scripts/pace/fit.py --group legs --no-opt_delay
#   python scripts/pace/fit.py --group arms,torso --warm_start logs/pace/s800_sim/<上一轮> --no-opt_delay
#   python scripts/pace/fit.py --group all --warm_start logs/pace/s800_sim/<上一轮> --sigma 0.05
# ② 的 warm_start 只是为了把腿的收敛值带进输出文件，臂/腰在它里面仍是中点，没有被热启动。
# 分组阶段 --no-opt_delay：delay 是全局共享参数，会通过冻结组把它们的误差泄漏进来。
#
# 为什么分组：各关节力矩尺度差 ~45 倍（腕 coul/τ_applied≈5.8%，髋只有 0.04%），
# 一套 bounds 尺度覆盖全部 109 维时必然牺牲一组 —— 实测全场统一 bounds 那轮，
# 轻关节 0.1% 准、髋/膝摩擦 400% 错。分组还能把维度从 109 降到 ~49。
# ⚠️ 验收指标看 TB 的 6_GroupResid/best_<组>（弧度 RMS），不是参数误差 ——
#    参数误差已被证明被 kp·RMS残差 卡死（见 pace_sysid_notes）。
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
                         "分组轮建议设 0 = 关掉：冻结组的常数偏移会把分母抬高、diff_score 被压低，"
                         "可能被误判为收敛而提前结束；0 时 diff_score < 0 恒不成立，等于关闭。")

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


def build_joint_groups(joint_order):
    """按运动链把关节分成 腿 / 腰 / 臂 三组，返回 {组名: [joint_order 下标]}。

    只按关节名分组，与参数无关 —— 每组的 armature/damping/friction/bias 一起优化。
    ⚠️ 新加的关节若落不进任何一组会直接报错，而不是被静默漏掉。
    """
    groups = {"legs": [], "torso": [], "arms": []}
    for i, name in enumerate(joint_order):
        short = name.split("_", 1)[1] if "_" in name else name  # "J00_HIP_PITCH_L" -> "HIP_PITCH_L"
        if any(k in short for k in ("HIP_", "KNEE_", "ANKLE_")):
            groups["legs"].append(i)
        elif short.startswith("TORSO"):
            groups["torso"].append(i)
        elif any(k in short for k in ("SHOULDER_", "ELBOW_", "WRIST_")):
            groups["arms"].append(i)
        else:
            raise ValueError(f"关节 {name} 落不进任何一组，请更新 build_joint_groups()")
    return groups


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
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

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    # ── 分组优化：解析要优化的组 + 载入 warm start ────────────────────────
    groups = build_joint_groups(joint_order)
    print("[group] 分组: " + ", ".join(f"{g}={len(v)}关节" for g, v in groups.items()))

    active_groups = [g.strip() for g in args_cli.group.split(",") if g.strip()]
    if "all" in active_groups:
        active_groups = None  # None = 全部（分组前的原行为）
    else:
        unknown = set(active_groups) - set(groups)
        if unknown:
            raise SystemExit(f"[group] 未知分组 {sorted(unknown)}，可用: all, {', '.join(groups)}")
        frozen = [g for g in groups if g not in active_groups]
        print(f"[group] 本轮只优化 {active_groups}，冻结 {frozen}"
              f"{'（未给 --warm_start 时冻结值 = bounds 中点）' if not args_cli.warm_start else ''}")
        # 提前停止判据在分组轮里会失灵：分数含冻结组的常数偏移 C，diff_score=(max-min)/min
        # 的分母被抬高，判据被压低，可能把「还没收敛」误判成收敛。分数本身与优化无关
        # （对全种群是同一个常数，排序不变），出问题的只有这个判据。
        _eps = env_cfg.sim2real.cmaes.epsilon if args_cli.epsilon is None else args_cli.epsilon
        if _eps and _eps > 0:
            print(f"[group] ⚠️ epsilon={_eps:g} 仍开着：分组轮的 diff_score 被冻结组的常数"
                  f"偏移压低，可能提前判收敛。长跑建议加 --epsilon 0 关掉。")

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

    opt = CMAESOptimizer(
        bounds=bounds_params,
        groups=groups,
        active_groups=active_groups,
        warm_start=warm_start,
        opt_delay=args_cli.opt_delay,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=env_cfg.sim2real.cmaes.epsilon if args_cli.epsilon is None else args_cli.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma if args_cli.sigma is None else args_cli.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
        segment_edges_hz=env_cfg.sim2real.cmaes.segment_edges_hz,
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

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
