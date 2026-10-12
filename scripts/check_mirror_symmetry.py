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

用法
----
    python scripts/check_mirror_symmetry.py

退出码 0 = 全部通过；1 = 有不一致。
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "assets_robot/engineai_S800/urdf/serial_s800_lock_wrist_torso.urdf"
COLLECTOR = REPO / "scripts/pace/data_collection.py"
CFG = REPO / "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/s800_pace_env_cfg.py"


def load_joint_order() -> list[str]:
    """从 env cfg 源码读出 joint_order。

    以前这里抄了一份 27 个名字的字面量，和 cfg 是两份会各自漂移的拷贝。
    cfg 用 ast 能直接读（和 test_experiments.py 的 test_actual_s800_partition 同一手法），
    这样关节表永远只有一处。
    """
    import ast

    tree = ast.parse(CFG.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == "joint_order")
    return ast.literal_eval(node.value)


JOINT_ORDER = load_joint_order()

MIRROR = np.diag([1.0, -1.0, 1.0])


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def world_axes(urdf_path: Path) -> dict[str, np.ndarray]:
    """参考位形 q=0 下，各关节转轴在世界坐标系中的方向。"""
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, dict] = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        rpy = np.array([float(x) for x in (origin.get("rpy") or "0 0 0").split()])
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

    walk("LINK_BASE", np.eye(3))
    return axes


def parse_collector(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Use the collector's shared nominal centers/amplitudes without launching Isaac."""
    sys.path.insert(0, str(path.parent))
    from s800_motion import nominal_motion

    center, amplitude = (x.numpy().astype(float) for x in nominal_motion(JOINT_ORDER))
    # Compatibility with the algebra used below: (wave+bias)*direction*scale.
    return np.sign(amplitude).tolist(), (center/amplitude).tolist(), np.abs(amplitude).tolist()


def main() -> int:
    axes = world_axes(URDF)
    dirs, bias, scale = parse_collector(COLLECTOR)

    if not (len(dirs) == len(bias) == len(scale) == len(JOINT_ORDER)):
        print(f"✗ 数组长度不匹配: dir={len(dirs)} bias={len(bias)} scale={len(scale)} 应为 {len(JOINT_ORDER)}")
        return 1

    index = {n: i for i, n in enumerate(JOINT_ORDER)}
    pairs: dict[tuple[str, str], str] = {}
    for name in JOINT_ORDER:
        m = re.match(r"(J\d+)_(.+)_([LR])$", name)
        if m:
            pairs[(m.group(2), m.group(3))] = name

    print(f"{'关节类型':18s} {'a_L 世界系':>22s} {'R·a_L':>22s} {'a_R 世界系':>22s} {'需要':>6s} {'实测':>6s} {'镜像':>6s}")
    print("-" * 108)

    failures = []
    # pairs 里 L / R 都是 key，只遍历 L 侧，否则同一类型会被处理两次（R 与自身比对必然失败）
    for (jtype, side), name_l in sorted(pairs.items()):
        if side != "L" or (jtype, "R") not in pairs:
            continue
        name_r = pairs[(jtype, "R")]
        a_l, a_r = axes[name_l], axes[name_r]
        mirror = MIRROR @ a_l

        if np.allclose(a_r, mirror, atol=2e-3):
            need = -1.0
        elif np.allclose(a_r, -mirror, atol=2e-3):
            need = +1.0
        else:
            print(f"{jtype:18s} ✗ 轴不满足镜像关系，无法判定（URDF 可能左右不对称）")
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

        print(f"{jtype:18s} {str(np.round(a_l,3)):>22s} {str(np.round(mirror,3)):>22s} "
              f"{str(np.round(a_r,3)):>22s} {need:+6.0f} {dirs[i_r]:+6.0f} {'✓' if ok else '✗':>6s}")

    print()
    if failures:
        print(f"✗ 未通过镜像对称: {', '.join(failures)}")
        return 1
    print("✓ 全部 13 对左右关节通过镜像对称校验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
