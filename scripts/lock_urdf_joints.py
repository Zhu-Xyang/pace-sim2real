#!/usr/bin/env python3
"""把一个 URDF 里指定的 revolute 关节改成 fixed，生成新的 URDF（原文件不动）。

为什么是「改成 fixed」而不是删掉关节：fixed 关节在 URDF→USD 转换时会被
merge_fixed_joints 合并进父 link（质量/惯量按关节位姿累加），于是该自由度从
articulation 里彻底消失、子 link 变成纯视觉 Xform。这正是 G1 的
g1_29dof_rev_1_0.urdf → g1_29dof_lock_waist_rev_1_0.urdf 用的约定：
只改 type，`<axis>` / `<limit>` 原样保留。

实现只替换 `<joint name="X" type="revolute">` 这一整行 —— 关节名在 mesh 路径、
注释里也可能出现，全局字符串替换会误伤。改完用 ElementTree 重新解析做自检：
关节类型、总质量、剩余 revolute 关节集合三项都对得上才写文件。

────────────────────────────────────────────────────────────────────────────
S800 完整重建流程（USD 被 .gitignore 忽略，换机器必须重跑这两步）

1) 改 URDF（本脚本）：

    python scripts/lock_urdf_joints.py \\
        --input  assets_robot/engineai_S800/urdf/serial_s800.urdf \\
        --output assets_robot/engineai_S800/urdf/serial_s800_lock_wrist_torso.urdf \\
        --preset s800_wrist_torso

   预期输出：可动关节 27 → 22，总质量守恒 79.901379 kg。

2) 转 USD（需要 Isaac Sim；`--fix-base --merge-joints` 之外的参数**全部是默认值**，
   与现有 usd/config.yaml 逐项核对过）：

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python /home/samsung/Humanoid/IsaacLab/scripts/tools/convert_urdf.py \\
        assets_robot/engineai_S800/urdf/serial_s800_lock_wrist_torso.urdf \\
        assets_robot/engineai_S800/usd_locked/serial_s800_lock_wrist_torso.usda \\
        --fix-base --merge-joints

3) 验收（顺序别换）：

    python scripts/joint_order_validation.py          # 仿真正好 22 个自由度
    python -m unittest discover -s scripts/pace -p 'test_*.py'
    python -u scripts/pace/run_s800_groups.py --check_only --num_envs 8 --fix_delay 3

   第 3 步的两个 floor test 都应打印 ✓ PASS（残差 ~1e-7 rad 量级）。
   它同时兜住「bounds 四段切片写错」这类静默错误。

⚠️ 改完必须重采数据：joint_names 变了，旧的 data/s800_sim/*.pt 会被
   validate_recording 直接拒绝；旧的 mean_*.pt 是 109 维，不能 warm start。
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ── S800 的分组锁定预设（--preset 用）─────────────────────────────────────
# 锁定理由见 docs/s800_single_part.md §8：
#   wrist_*  扫频把它们驱进共振（指令 ±0.05 rad、实际 ±0.5 rad），armature 不可辨识
#   TORSO_YAW 手臂反作用力矩带动它，污染上半身辨识
PRESETS = {
    "s800_wrist_torso": [
        "J12_TORSO_YAW",
        "J18_WRIST_PITCH_L", "J19_WRIST_ROLL_L",
        "J32_WRIST_PITCH_R", "J33_WRIST_ROLL_R",
    ],
}


def total_mass(root: ET.Element) -> float:
    return sum(float(link.find("inertial/mass").get("value")) for link in root.findall("link"))


def revolute_joints(root: ET.Element) -> set[str]:
    return {j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="原 URDF（不会被修改）")
    parser.add_argument("--output", type=Path, required=True, help="新 URDF 的输出路径")
    parser.add_argument("--lock", nargs="*", default=[], help="要锁定的关节名")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="用预设的关节列表（会与 --lock 合并）")
    args = parser.parse_args()

    targets = list(dict.fromkeys([*PRESETS.get(args.preset, []), *args.lock]))
    if not targets:
        parser.error("至少给一个 --lock 或 --preset")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output 不能和 --input 相同；原文件必须保持不变")

    before = ET.parse(args.input).getroot()
    mass_before = total_mass(before)
    rev_before = revolute_joints(before)
    known = {j.get("name") for j in before.findall("joint")}
    missing = [n for n in targets if n not in known]
    if missing:
        parser.error(f"URDF 里没有这些关节: {missing}")

    lines = args.input.read_text().splitlines(keepends=True)
    for name in targets:
        # 整行匹配，避免误伤 mesh 路径/注释里出现的同名子串
        pattern = re.compile(rf'^(\s*<joint\s+name="{re.escape(name)}"\s+type=")revolute(")')
        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        if len(hits) != 1:
            print(f"[FAIL] {name}: 期望正好 1 处 <joint> 声明，实际 {len(hits)} 处", file=sys.stderr)
            return 1
        i = hits[0]
        old = lines[i]
        lines[i] = pattern.sub(r"\1fixed\2", old)
        print(f"  {name:<22} line {i + 1}: {old.strip()}  →  {lines[i].strip()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines))

    # ── 自检 ──────────────────────────────────────────────────────────────
    after = ET.parse(args.output).getroot()
    mass_after = total_mass(after)
    rev_after = revolute_joints(after)
    problems = []
    fixed_after = {j.get("name") for j in after.findall("joint") if j.get("type") == "fixed"}
    for name in targets:
        if name not in fixed_after:
            problems.append(f"{name} 不是 fixed")
    if rev_before - set(targets) != rev_after:
        problems.append(f"revolute 关节集合不符: 少了 {rev_before - set(targets) - rev_after}，"
                        f"多了 {rev_after - (rev_before - set(targets))}")
    if abs(mass_after - mass_before) > 1e-9:
        problems.append(f"总质量变了: {mass_before:.6f} → {mass_after:.6f} kg (锁定不应改变总质量)")
    if len(after.findall("link")) != len(before.findall("link")):
        problems.append("link 数量变了（本脚本只应改 joint type）")

    print(f"\n输入 {args.input}")
    print(f"  可动关节 {len(rev_before)} → {len(rev_after)}   总质量 {mass_before:.6f} kg")
    if problems:
        args.output.unlink()
        print("[FAIL] 自检未通过，已删除输出文件：", file=sys.stderr)
        for p in problems:
            print(f"  · {p}", file=sys.stderr)
        return 1
    print(f"输出 {args.output}")
    print(f"  ✓ 自检通过：{len(targets)} 个关节已 fixed，revolute 剩 {len(rev_after)} 个，"
          f"总质量守恒 {mass_after:.6f} kg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
