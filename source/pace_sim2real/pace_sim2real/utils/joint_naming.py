# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""关节命名工具。

S800 的关节名形如 `J00_HIP_PITCH_L` / `J12_TORSO_YAW`，
「关节类型」= 去掉 `Jxx_` 前缀和 `_L`/`_R` 后缀的部分
（`J00_HIP_PITCH_L` → `HIP_PITCH`，`J12_TORSO_YAW` → `TORSO_YAW`）。

按类型（而不是按位置）写 per-joint 参数表，可以让表既好 grep、又不会和
`joint_order` 静默错位 —— 关节表增删时位置索引会平移，类型表不会。
"""

from __future__ import annotations


def joint_type(name: str) -> str:
    """`J00_HIP_PITCH_L` → `HIP_PITCH`；`J12_TORSO_YAW` → `TORSO_YAW`。"""
    rest = name.split("_", 1)[1] if name.startswith("J") and "_" in name else name
    for side in ("_L", "_R"):
        if rest.endswith(side):
            return rest[: -len(side)]
    return rest
