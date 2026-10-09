# scripts/joint_order_validation.py
"""启一次无头仿真，校验 env cfg 与资产是否对得上。

检查项：
  1) config 的 joint_order 每个名字都能在 articulation.joint_names 里找到（否则 fit 会崩）
  2) Isaac Lab 的真实 DOF 顺序 vs config 顺序的映射
  3) 每个关节是否都被某个 actuator drive 覆盖（漏掉的关节不会有 PD 控制）
  4) 关节限位（rad）、初始位置、sim dt、基座是否焊死
  5) 跑一步 step 确认不炸

用法:
  python scripts/joint_order_validation.py --task Isaac-Pace-G1-v0
  python scripts/joint_order_validation.py --task Isaac-Pace-S800-v0
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate joint_order / actuators against the asset.")
parser.add_argument("--task", type=str, default="Isaac-Pace-S800-v0", help="Task name to validate.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import pace_sim2real.tasks  # noqa: F401,E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg)
articulation = env.unwrapped.scene["robot"]
joint_names = list(articulation.joint_names)
joint_order = env_cfg.sim2real.joint_order

print(f"\n=== {args_cli.task} ===")
print(f"Isaac Lab DOF 数: {len(joint_names)}   config joint_order 长度: {len(joint_order)}")
print(f"sim.dt = {env.unwrapped.sim.cfg.dt}  decimation = {env_cfg.decimation}")

print("\n=== [1/2] 关节顺序映射 ===")
missing = [n for n in joint_order if n not in joint_names]
if missing:
    print(f"  ❌ joint_order 里有 {len(missing)} 个名字在 articulation 里不存在: {missing}")
else:
    print(f"  ✓ joint_order 的 {len(joint_order)} 个名字全部存在")
for i, name in enumerate(joint_names):
    tag = ""
    if name in joint_order:
        tag = f"  ← joint_order[{joint_order.index(name)}]"
    else:
        tag = "  ← ⚠️ 不在 joint_order 里（不会被辨识）"
    print(f"  [{i:2d}] {name}{tag}")

print("\n=== [3] actuator 覆盖 ===")
cover = {}
for drive_name, act in articulation.actuators.items():
    idx = act.joint_indices
    if isinstance(idx, slice):
        names = joint_names[idx]
    else:
        names = [joint_names[i] for i in idx]
    cover[drive_name] = names
    print(f"  drive '{drive_name}': {len(names)} 个关节")
uncovered = [n for n in joint_names if not any(n in v for v in cover.values())]
print(f"  未被任何 drive 覆盖: {uncovered or '无 ✓'}")

print("\n=== [4] 限位 / 初始状态 ===")
lim = articulation.data.joint_pos_limits[0].cpu()
pos = articulation.data.joint_pos[0].cpu()
print(f"  {'joint':<34}{'lower':>9}{'upper':>9}{'init pos':>10}")
for i, name in enumerate(joint_names):
    print(f"  {name:<34}{lim[i,0]:>9.3f}{lim[i,1]:>9.3f}{pos[i]:>10.4f}")
print(f"  根 prim: {articulation.root_physx_view.prim_paths[0] if hasattr(articulation, 'root_physx_view') else 'n/a'}")
print(f"  基座位置: {articulation.data.root_pos_w[0].cpu().tolist()}")

print("\n=== [5] 空动作跑一步 ===")
env.reset()  # gymnasium 要求先 reset
for drive_name, act in articulation.actuators.items():
    kp = act.stiffness[0].cpu() if act.stiffness is not None else None
    kd = act.damping[0].cpu() if act.damping is not None else None
    if kp is not None:
        print(f"  drive '{drive_name}' kp 范围 [{kp.min():.2f}, {kp.max():.2f}]"
              f"  kd 范围 [{kd.min():.3f}, {kd.max():.3f}]")
actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
obs, _, _, _, _ = env.step(actions)
_terms = list(obs["policy"].keys()) if isinstance(obs, dict) and "policy" in obs else "n/a"
print(f"  ✓ step 正常返回，obs groups {list(obs.keys()) if isinstance(obs, dict) else 'n/a'}"
      f"，policy 项 {_terms}")
print(f"  力矩绝对值最大: {articulation.data.applied_torque.abs().max().item():.3f} Nm")

print("\n校验完成。")
env.close()
simulation_app.close()
