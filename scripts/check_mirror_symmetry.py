#!/usr/bin/env python3
"""校验 data_collection.py 的激励是否镜像对称。

背景
----
PACE 原文对**悬吊真机**采集要求对称指令以抵消净基座力旋量（固定基座时不需要）。
但即使不追求净力旋量抵消，"左右是否镜像"也是一个可客观校验的设计属性 ——
写错符号会让左右肢体反相，视觉上不易察觉，却会改变激励结构。

判据
----
矢状面反射 M = diag(1, -1, 1)。设 a_L / a_R 为左右关节在参考位形 q=0 时
**世界坐标系**下的转轴（由 URDF 的 origin rpy 沿链累积得到）：

    a_R == +M·a_L   →   需要 dir_R = -1    （轴落在镜面内：roll / yaw）
    a_R == -M·a_L   →   需要 dir_R = +1    （轴沿镜面法向：pure pitch）

依据：反射是反向的，绕 a_L 转 θ 镜像后等于绕 M·a_L 转 -θ。

⚠️ 常见错误判据是「两个轴向量是否相同」—— 那只在轴落在镜面内时成立。
   纯 pitch 轴（如 (0,1,0)）满足 M·a_L = -a_L，两向量"看起来一样"，
   但它恰恰是需要 dir_R = +1 的镜像情形。
⚠️ 每台机器人的转轴不同，方向数组**不能照抄**（G1 的肩偏航/腕部就与 S800 相反，
   且 G1 的髋轴是斜的 [0.98,0,0.17]，不能按名字猜）。

用法
----
    python scripts/check_mirror_symmetry.py --robot s800
    python scripts/check_mirror_symmetry.py --robot g1

退出码 0 = 全部通过；1 = 有不一致。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
COLLECTOR = REPO / "scripts/pace/data_collection.py"
MIRROR = np.diag([1.0, -1.0, 1.0])

# 机器人配置：ROBOT_TABLES 里的键 / URDF / joint_order / 基座 link
# 关节表与方向数组都取自 scripts/pace/robot_tables.py（单一数据源）——
# 以前这里各存一份，改了真值/激励不同步就会静默校验通过。
sys.path.insert(0, str(REPO / "scripts/pace"))
from robot_tables import JOINT_ORDER, ROBOT_TABLES  # noqa: E402

# 校验器专用的信息：URDF 路径 + 哪个 link 是（固定的）基座
ROBOTS = {
    "s800": {
        "key": "s800_sim",
        "urdf": REPO / "assets_robot/engineai_S800/urdf/serial_s800.urdf",
        "root": "LINK_BASE",
    },
    "g1": {
        "key": "g1_sim",
        "urdf": REPO / "assets_robot/g1_description/g1_29dof_rev_1_0.urdf",
        "root": "pelvis",
    },
}



def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def world_axes(urdf_path: Path, root_link: str) -> dict[str, np.ndarray]:
    """参考位形 q=0 下，各关节转轴在世界坐标系中的方向。"""
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, dict] = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        rpy = np.array([float(x) for x in (origin.get("rpy") or "0 0 0").split()]) if origin is not None else np.zeros(3)
        ax = j.find("axis")
        axis = np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else np.array([0.0, 0.0, 1.0])
        joints[j.get("name")] = {
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "rpy": rpy,
            "axis": axis,
        }
    children: dict[str, list[str]] = {}
    for name, d in joints.items():
        children.setdefault(d["parent"], []).append(name)

    axes: dict[str, np.ndarray] = {}

    def walk(link: str, rot: np.ndarray) -> None:
        for joint_name in children.get(link, []):
            d = joints[joint_name]
            rot_j = rot @ rpy_to_matrix(d["rpy"])
            axes[joint_name] = rot_j @ d["axis"]
            walk(d["child"], rot_j)

    walk(root_link, np.eye(3))
    return axes


def split_side(name: str):
    """'J00_HIP_PITCH_L' -> ('HIP_PITCH','L')；'left_hip_pitch_joint' -> ('hip_pitch_joint','L')"""
    m = re.match(r"^J\d+_(.+)_([LR])$", name)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(left|right)_(.+)$", name)
    if m:
        return m.group(2), ("L" if m.group(1) == "left" else "R")
    return None, None


def parse_collector(robot_key: str):
    """取该机器人的 directions / bias / scale（来自 robot_tables.py 的单一数据源）。

    以前这里是 AST 解析 data_collection.py —— 表搬进 robot_tables.py 后直接读模块即可。
    """
    table = ROBOT_TABLES.get(robot_key)
    if table is None:
        raise SystemExit(f"✗ robot_tables.py 里没有 {robot_key}（可用：{sorted(ROBOT_TABLES)}）")
    return table["directions"], table["bias"], table["scale"]


def main() -> int:
    ap = argparse.ArgumentParser(description="镜像对称校验")
    ap.add_argument("--robot", default="s800", choices=sorted(ROBOTS))
    args = ap.parse_args()
    cfg = ROBOTS[args.robot]
    JOINT_ORDER_LOCAL = JOINT_ORDER[cfg["key"]]

    axes = world_axes(cfg["urdf"], cfg["root"])
    dirs, bias, scale = parse_collector(cfg["key"])

    if not (len(dirs) == len(bias) == len(scale) == len(JOINT_ORDER_LOCAL)):
        print(f"✗ 数组长度不匹配: dir={len(dirs)} bias={len(bias)} scale={len(scale)} 应为 {len(JOINT_ORDER_LOCAL)}")
        return 1

    index = {n: i for i, n in enumerate(JOINT_ORDER_LOCAL)}
    pairs: dict[tuple[str, str], str] = {}
    for name in JOINT_ORDER_LOCAL:
        jtype, side = split_side(name)
        if jtype:
            pairs[(jtype, side)] = name

    print(f"[{args.robot}] 基座={cfg['root']}  关节 {len(JOINT_ORDER_LOCAL)} 个")
    print(f"{'关节类型':22s} {'a_L 世界系':>22s} {'R·a_L':>22s} {'a_R 世界系':>22s} {'需要':>6s} {'实测':>6s} {'镜像':>6s}")
    print("-" * 112)

    failures = []
    n_pairs = 0
    # pairs 里 L / R 都是 key，只遍历 L 侧，否则同一类型会被处理两次（R 与自身比对必然失败）
    for (jtype, side), name_l in sorted(pairs.items()):
        if side != "L" or (jtype, "R") not in pairs:
            continue
        n_pairs += 1
        name_r = pairs[(jtype, "R")]
        if name_l not in axes or name_r not in axes:
            print(f"{jtype:22s} ✗ URDF 里找不到 {name_l} / {name_r}")
            failures.append(jtype)
            continue
        a_l, a_r = axes[name_l], axes[name_r]
        mirror = MIRROR @ a_l

        if np.allclose(a_r, mirror, atol=2e-3):
            need = -1.0
        elif np.allclose(a_r, -mirror, atol=2e-3):
            need = +1.0
        else:
            print(f"{jtype:22s} ✗ 轴不满足镜像关系，无法判定（URDF 可能左右不对称）")
            failures.append(jtype)
            continue

        i_l, i_r = index[name_l], index[name_r]
        c = np.linspace(-1.0, 1.0, 201)
        q_l = (c + bias[i_l]) * dirs[i_l] * scale[i_l]
        q_r = (c + bias[i_r]) * dirs[i_r] * scale[i_r]
        symmetric = np.allclose(q_r, need * q_l, atol=1e-9)
        dir_ok = abs(dirs[i_r] - need) < 1e-9
        ok = symmetric and dir_ok
        if not ok:
            failures.append(jtype)

        print(f"{jtype:22s} {str(np.round(a_l,3)):>22s} {str(np.round(mirror,3)):>22s} "
              f"{str(np.round(a_r,3)):>22s} {need:+6.0f} {dirs[i_r]:+6.0f} {'✓' if ok else '✗':>6s}")

    print()
    if failures:
        print(f"✗ 未通过镜像对称: {', '.join(failures)}")
        return 1
    print(f"✓ 全部 {n_pairs} 对左右关节通过镜像对称校验（--robot {args.robot}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
