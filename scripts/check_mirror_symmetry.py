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
URDF = REPO / "assets_robot/engineai_S800/urdf/serial_s800.urdf"
COLLECTOR = REPO / "scripts/pace/data_collection.py"

JOINT_ORDER = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R", "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L", "J18_WRIST_PITCH_L", "J19_WRIST_ROLL_L",
    "J27_SHOULDER_PITCH_R", "J28_SHOULDER_ROLL_R", "J29_SHOULDER_YAW_R",
    "J30_ELBOW_PITCH_R", "J31_ELBOW_YAW_R", "J32_WRIST_PITCH_R", "J33_WRIST_ROLL_R",
]

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
    """从 data_collection.py 取出 directions / bias / scale（跳过注释行）。"""
    src = "\n".join(l for l in path.read_text().split("\n") if not l.strip().startswith("#"))

    def grab(name: str) -> list[float]:
        i = src.index(f"{name} = torch.tensor(")
        j = src.index("device=", i)
        body = src[i:j]
        body = body[body.index("[") + 1 : body.rindex("]")]
        body = re.sub(r"#.*", "", body)
        return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*", body)]

    return grab("trajectory_directions"), grab("trajectory_bias"), grab("trajectory_scale")


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
