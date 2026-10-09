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
                             torch.tensor([float(_gt["delay"])])]).to(bounds_params.dtype)
        _eps = 1e-6 * _gt_vec.abs().clamp_min(1e-6)
        bounds_params = torch.stack([_gt_vec - _eps, _gt_vec + _eps], dim=1).to(bounds_params.device)
        print(f"[floor_test] bounds 已收缩到数据里的注入真值 ±1e-6 相对，只跑 1 代"
              f"（真值 delay={float(_gt['delay']):g} 步）")

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
        max_iteration=1 if args_cli.floor_test else env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=0.0 if args_cli.floor_test else (
            env_cfg.sim2real.cmaes.epsilon if args_cli.epsilon is None else args_cli.epsilon),
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

    if args_cli.floor_test:
        _s = opt.scores_buffer[0]                     # 第 1 代全种群的分数
        _best, _worst = _s.min().item(), _s.max().item()
        _n_j = len(joint_order)
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
        _res = (_sim_best - _real).abs()
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
        print("              判读：开局就在 1e-8 量级 ⇒ **口径一致**；之后被冲击抬高的部分"
              "来自非光滑事件")
        print("              （限位冲击/自碰撞/自激），要靠改激励设计解决，不是改代码。")

        # 判定看 min：口径一致时**存在**一组参数能逐位复现数据。用 max 判会被整数量化
        # 误伤 —— 框收缩到真值 ±1e-6 时，若真值压在延时格子边界上，一半样本会取到相邻的
        # 整数延时，损失差好几个数量级（那是取整方式的问题，不是口径不一致）。
        if _best < 1e-10:
            print("    ✓ PASS —— 采集与拟合口径一致（时间对齐/零偏/延时/初始状态）")
            if _at_floor < _s.numel():
                print(f"    （{_s.numel() - _at_floor} 个样本没到地板：多半是延时落在整数格子"
                      f"边界上被取到相邻值，或系统本身有自激/强非线性。看 min 即可。）")
        elif _early < 1e-6:
            print("    ✓ 口径一致（开局在浮点地板）—— 但轨迹里有非光滑事件，全局阈值到不了。")
            print("      这不是流程的问题：真值参数已经能复现数据，只是某处有接触/限位冲击。")
            print("      ⇒ 可以继续拟合；受影响的是那几个被冲击的关节。要拿到干净地板就改激励。")
        else:
            print("    ✗ FAIL —— 两边口径有系统性差异（开局就没到地板），先别往下拟合。常见原因：")
            print("      · 数据是用**别的 cfg** 采的（增益/URDF/sim_dt 改过之后没重采）")
            print("      · 初始状态或时间对齐差一步（des_dof_pos 与 dof_pos 错位）")
            print("      · 代码里的真值常量与注入值不一致（本检查读的是数据里的 gt，不是常量）")
            raise SystemExit(1)
        print("=" * 72)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
