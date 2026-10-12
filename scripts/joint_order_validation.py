# scripts/joint_order_validation.py
"""核对仿真里的关节表与 env cfg 的 joint_order 是否严格一致。

改资产（锁关节 / 重转 USD）之后必跑。三件事：
  1) Isaac 侧的 joint_names 逐项打印，便于人工核对
  2) cfg 的 joint_order 每一项都能在 Isaac 侧 index 到
  3) Isaac 侧的自由度数目正好等于 len(joint_order) —— 锁定的关节必须**真的消失**，
     而不是以 fixed 类型留在 articulation 里（actuator 的正则是「key→关节」单向校验，
     那种情况不会报错，只会静默拿到 USD 默认的增益/限幅）
"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True)
simulation_app = app.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import pace_sim2real.tasks  # noqa: E402,F401  注册环境

# S800PaceEnvCfg 的 cfg 是必填参数，不能裸 gym.make
env = gym.make("Isaac-Pace-S800-v0", cfg=parse_env_cfg("Isaac-Pace-S800-v0", device="cuda:0", num_envs=1))
articulation = env.unwrapped.scene["robot"]
joint_order = env.unwrapped.cfg.sim2real.joint_order

print("\n=== Isaac Lab 关节顺序 ===")
for i, name in enumerate(articulation.joint_names):
    print(f"  [{i:2d}] {name}")

print("\n=== Config joint_order ===")
for i, name in enumerate(joint_order):
    isaac_idx = articulation.joint_names.index(name)
    print(f"  [{i:2d}] {name}  →  Isaac Lab index: {isaac_idx}")

n_sim, n_cfg = articulation.num_joints, len(joint_order)
print(f"\n=== 核对 ===")
print(f"  仿真自由度   : {n_sim}")
print(f"  cfg 关节数   : {n_cfg}")

problems = []
if n_sim != n_cfg:
    problems.append(f"自由度数目不一致：仿真 {n_sim} != cfg {n_cfg}。"
                    "多出来的通常是「改成 fixed 了但没被 merge 掉」的关节。")
extra = [n for n in articulation.joint_names if n not in set(joint_order)]
if extra:
    problems.append(f"仿真里有 cfg 未列出的关节: {extra}")
missing = [n for n in joint_order if n not in set(articulation.joint_names)]
if missing:
    problems.append(f"cfg 列了但仿真里没有的关节: {missing}")
if sorted(joint_order) != sorted(articulation.joint_names):
    problems.append("两边关节集合不一致（顺序由各自索引决定，集合必须相同）")

if problems:
    print("\n  ✗ FAIL")
    for p in problems:
        print(f"    · {p}")
    code = 1
else:
    print(f"\n  ✓ PASS —— {n_cfg} 个关节一一对应，锁定关节已从仿真里消失")
    code = 0

env.close()
simulation_app.close()
raise SystemExit(code)
