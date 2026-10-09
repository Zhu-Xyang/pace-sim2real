# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Name of the task.")
parser.add_argument("--data", type=str, default=None,
                    help="数据文件（相对 data/ 或绝对路径）。默认用 env cfg 的 data_dir。"
                         "分组激励采的数据另存为 chirp_data_<组>.pt，用这里指定。")

# ── 分组优化（手 / 腿 / 腰）──────────────────────────────────────────────
# 每轮只优化一个组，其余组冻结在 --warm_start 给出的值上（不给就从 bounds 中点起步）。
#
# URDF 是三条挂在 LINK_BASE 上的独立支链（左腿 / 右腿 / 腰→双臂），基座焊死
# （fix_root_link=True）⇒ 质量阵支链间块对角、损失可加：腿的结果不受臂/腰冻结值影响。
# 但支链内耦合：torso 是双臂的父关节、臂对 torso 有反作用 ⇒ 这两个必须同一轮。
#
# 从零开始的顺序（不给 warm_start ⇒ 本组从中点全空间搜索）：
#   python scripts/pace/fit.py --group legs
#   python scripts/pace/fit.py --group arms,torso --warm_start logs/pace/s800_sim/<上一轮>
#   python scripts/pace/fit.py --group all --warm_start logs/pace/s800_sim/<上一轮> --sigma 0.05
# ② 的 warm_start 只是为了把腿的收敛值带进输出文件，臂/腰在它里面仍是中点，没有被热启动。
# delay 是全局共享参数：损失默认只算被拟合的关节（--loss fitted），所以它由本组数据决定，
# 不会像以前那样把冻结组的误差泄漏进来 ⇒ 分组轮不用再关 --opt_delay。
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
parser.add_argument("--loss", type=str, default="fitted", choices=("fitted", "all"),
                    help="损失在哪些关节上算。fitted（默认）= 只在被优化的关节上算；"
                         "all = 全部关节（分组前的原行为，分数会被未辨识关节垫高、不可比）。"
                         "⚠️ delay 是全局共享参数，all 会让冻结组的误差通过它泄漏进来。")
parser.add_argument("--armature_param", type=str, default="log", choices=("log", "linear"),
                    help="armature 的参数化。log（默认）= 按数量级搜索，warm_start 与初值都落在"
                         "几何中点；linear = 直接线性（老行为，真值容易落在区间角落里 → 起点偏重"
                         "→ 掉进延迟陷阱）。")
parser.add_argument("--floor_test", action="store_true",
                    help="一致性检查：把 bounds 收缩到数据里的真值 ±1e-6，只跑 1 代。"
                         "采集与拟合若在时间对齐/零偏/延时/初始状态上完全一致，损失应 ≈0"
                         "（<1e-10）。这是辨识流程最重要的一步自检 —— 不通过就别往下跑。")
parser.add_argument("--fix_delay", type=int, default=None,
                    help="把延时钉死在这个整数步上、不参与优化。延时是整数、损失对它是阶梯状，"
                         "一旦滑进「用更短延时补相位」的谷就回不来；钉住后逐个扫 2/3/4/5 步"
                         "比最终损失，最低的那个就是真延时（也是真机上确认延时的办法）。")

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

# 关节分组规则（腿/腰/臂）在 robot_tables.py —— 与 data_collection/check_mirror_symmetry
# 共用同一份，避免各自维护一套关键词表而漂移。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_tables import GROUP_RULES, joint_group, joints_in_group  # noqa: E402


def build_joint_groups(joint_order):
    """按运动链把关节分成 腿 / 腰 / 臂 三组，返回 {组名: [joint_order 下标]}。

    规则见 robot_tables.GROUP_RULES。只按关节名分组，与参数无关 —— 每组的
    armature/damping/friction/bias 一起优化。
    ⚠️ 落不进任何一组的关节会直接报错（robot_tables.joint_group），而不是被静默漏掉。

    分组能成立的前提是「支链之间动力学解耦」：基座焊死 + 三条挂在基座上的独立支链
    （左腿 / 右腿 / [腰→双臂]）。所以腿和臂可以分开优化，但**腰和臂必须同一轮**
    （腰是臂的父关节、臂对腰有反作用）→ 用 `--group arms,torso`（别名 upper）。
    """
    groups = {"legs": [], "torso": [], "arms": []}
    for i, name in enumerate(joint_order):
        groups[joint_group(name)].append(i)
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

    _data_arg = args_cli.data or env_cfg.sim2real.data_dir
    data_file = Path(_data_arg)
    if not data_file.is_absolute():
        data_file = project_root() / "data" / data_file
    if not data_file.exists():
        raise SystemExit(f"[fit] 数据文件不存在: {data_file}")
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    # ── 一致性检查（floor test）：把 bounds 收缩到真值本身 ──────────────────
    # 采集和拟合若在时间对齐 / 零偏 / 延时 / 初始状态上完全一致，把真值原样放进去，
    # 损失应该只剩浮点噪声（<1e-10）。对不上就说明两边口径有系统性差异 ——
    # 这时任何拟合结果都不可信，先修流程。
    floor_gt = None
    if args_cli.floor_test:
        if "gt" not in data:
            raise SystemExit("[floor_test] 数据文件里没有 'gt' 字段（老数据），无法把 bounds 收缩到真值。")
        _gt = data["gt"]
        if _gt["joint_order"] != list(joint_order):
            raise SystemExit("[floor_test] 数据里的 joint_order 与 env cfg 不一致，先对齐再跑。")
        floor_gt = torch.cat([_gt["armature"].flatten(), _gt["damping"].flatten(),
                              _gt["friction"].flatten(), _gt["bias"].flatten(),
                              torch.tensor([float(_gt["delay"])])]).to(bounds_params.dtype).cpu()
        _eps = 1e-6 * floor_gt.abs().clamp_min(1e-6)
        bounds_params = torch.stack([floor_gt - _eps, floor_gt + _eps], dim=1).to(bounds_params.device)
        print(f"[floor_test] bounds 已收缩到注入真值 ±1e-6 相对，只跑 1 代。"
              f"（真值 delay={float(_gt['delay'])} 步）")

    # 延时钉死在整数步上（见 --fix_delay 的说明）
    if args_cli.fix_delay is not None:
        _k = float(args_cli.fix_delay)
        _max_delay = getattr(env_cfg.scene.robot.actuators.get("joints"), "max_delay", None)
        if _max_delay is not None and _k > _max_delay + 1:
            raise SystemExit(
                f"[fix_delay] {_k:g} 步超出 DelayBuffer 允许的范围 0..{_max_delay + 1}"
                f"（cfg 的 max_delay={_max_delay}），会直接 ValueError。")
        bounds_params[4 * len(joint_order)] = torch.tensor([_k, _k], dtype=bounds_params.dtype,
                                                           device=bounds_params.device)
        args_cli.opt_delay = False
        print(f"[fix_delay] 延时钉死在 {_k:g} 步（不参与优化）")

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
        # "upper" 是 torso+arms 的别名（G1 sim2sim 手册的叫法）
        if "upper" in active_groups:
            active_groups = [g for g in active_groups if g != "upper"] + ["torso", "arms"]
            print("[group] 'upper' 展开为 torso+arms（腰是臂的父关节，必须同一轮）")
        unknown = set(active_groups) - set(groups)
        if unknown:
            raise SystemExit(f"[group] 未知分组 {sorted(unknown)}，可用: all, {', '.join(groups)}, upper")
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

    # ── 损失关节：默认只算被优化的关节 ────────────────────────────────────
    loss_joints = None
    if active_groups is not None and args_cli.loss == "fitted":
        loss_joints = sorted({j for g in active_groups for j in groups[g]})
    # 数据里标了「哪些关节被激励」时对一下：没被激励的关节不该进拟合（它们的运动
    # 只反映重力塌陷，不含激励信息），错配会让拟合去解释一段无信息的轨迹。
    _excited = data.get("excited")
    if _excited is not None:
        _ex_idx = {joint_order.index(n) for n in _excited if n in joint_order}
        _fit_idx = set(loss_joints) if loss_joints is not None else set(range(len(joint_order)))
        if _fit_idx - _ex_idx:
            _bad = [joint_order[i] for i in sorted(_fit_idx - _ex_idx)]
            print(f"[warn] 要拟合的关节里有 {len(_bad)} 个没被激励"
                  f"（数据只激励了 {len(_ex_idx)} 个）：{_bad[:6]}{' ...' if len(_bad) > 6 else ''}")
            print("[warn] 这些关节的轨迹只含重力塌陷、不含激励信息，建议 --group 与采集的 --excite 对齐。")

    opt = CMAESOptimizer(
        bounds=bounds_params,
        groups=groups,
        active_groups=active_groups,
        warm_start=warm_start,
        opt_delay=args_cli.opt_delay,
        log_armature=(args_cli.armature_param == "log"),
        loss_joints=loss_joints,
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
        _n_j = len(loss_joints) if loss_joints is not None else len(joint_order)
        _rms = (_best / max(_n_j, 1)) ** 0.5          # 折成逐关节 RMS 残差 [rad]
        _at_floor = int((_s < 1e-10).sum())
        print("\n" + "=" * 72)
        print(f"[floor_test] 真值原样放进 {_s.numel()} 个环境，只跑 1 代：")
        print(f"    损失  min {_best:.3e}  max {_worst:.3e}   （{_n_j} 个关节的平方误差和，rad²）")
        print(f"    折算  逐关节 RMS 残差 {_rms:.3e} rad")
        print(f"    达到浮点地板(<1e-10) 的样本：{_at_floor}/{_s.numel()}")
        # 判定看 min：口径一致时**存在**一组参数能逐位复现数据。用 max 判会被整数量化
        # 误伤 —— 框收缩到真值 ±1e-6 时，若真值压在延时格子边界上，一半样本会取到
        # 相邻的整数延时，损失差好几个数量级（那是取整方式的问题，不是口径不一致）。
        if _best < 1e-10:
            print("    ✓ PASS —— 采集与拟合口径一致（时间对齐/零偏/延时/初始状态）")
            if _at_floor < _s.numel():
                print(f"    （{_s.numel() - _at_floor} 个样本没到地板：多半是延时落在整数格子"
                      f"边界上被取到相邻值。看 min 即可。）")
        else:
            print("    ✗ FAIL —— 两边口径有系统性差异，先别往下拟合。常见原因：")
            print("      · 采集与拟合用了不同的 URDF / 增益 / sim_dt")
            print("      · 初始状态或时间对齐差一步（des_dof_pos 与 dof_pos 错位）")
            print("      · delay 不在搜索范围内（--fix_delay 给错）或被夹到边界")
            raise SystemExit(1)
        print("=" * 72)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
