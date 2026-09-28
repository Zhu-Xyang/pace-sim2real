# scripts/verify_joint_order.py
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True)
sim = app.app

import gymnasium as gym
import pace_sim2real.tasks  # 注册环境

env = gym.make("Isaac-Pace-S800-v0")
articulation = env.unwrapped.scene["robot"]

print("\n=== Isaac Lab 关节顺序 ===")
for i, name in enumerate(articulation.joint_names):
    print(f"  [{i:2d}] {name}")

print("\n=== Config joint_order ===")
joint_order = env.unwrapped.cfg.sim2real.joint_order
for i, name in enumerate(joint_order):
    isaac_idx = articulation.joint_names.index(name)
    print(f"  [{i:2d}] {name}  →  Isaac Lab index: {isaac_idx}")

env.close()
simulation_app.close()
