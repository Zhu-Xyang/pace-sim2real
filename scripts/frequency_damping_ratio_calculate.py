#!/usr/bin/env python3
"""S800 关节固有频率与阻尼比计算工具

物理模型（PhysX 求解的二阶系统）:

    I_total * q_ddot + (d + kd) * q_dot + kp * q = tau_cmd

其中:
    I_total = Ia + I_link  (armature 惯量 + 下游连杆等效惯量)
    d       = viscous_friction (电机粘性摩擦系数)
    kd      = PD 控制器微分增益
    kp      = PD 控制器比例增益

计算公式:
    omega_n = sqrt(kp / I) / (2*pi)   [Hz]
    zeta    = (d + kd) / (2 * sqrt(kp * I))

用法:
    python frequency_damping_ratio_calculate.py                  # 默认: 同时显示 Ia 和 I_total
    python frequency_damping_ratio_calculate.py --mode ia         # 只用纯 armature
    python frequency_damping_ratio_calculate.py --mode itotal     # 只用 I_total
    python frequency_damping_ratio_calculate.py --mode both      # 对比两者 (默认)
    python frequency_damping_ratio_calculate.py --json           # JSON 输出
    python frequency_damping_ratio_calculate.py --csv             # CSV 输出

参数来源:
    Ia, kp, kd  -> kpkd.py _TYPE_TABLE (可在此文件顶部直接修改)
    d           -> s800_pace_env_cfg.py viscous_friction 字典
    I_link      -> URDF 下游连杆等效惯量 (damping_ratio_analysis.md)
"""

import math
import json
import argparse
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# 参数定义 (来自 kpkd.py _TYPE_TABLE 和 s800_pace_env_cfg.py)
# 修改这里的值即可重新计算，无需改代码
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class JointParams:
    """单个关节类型的物理参数."""
    name: str        # 关节类型名
    Ia: float        # armature 惯量 [kg·m²] (来自 kpkd.py)
    kp: float        # PD 比例增益 [Nm/rad] (来自 kpkd.py)
    kd: float        # PD 微分增益 [Nm·s/rad] (来自 kpkd.py)
    d: float         # 粘性摩擦系数 [Nm·s/rad] (来自 env_cfg viscous_friction)
    I_link: float    # 下游连杆等效惯量 [kg·m²] (来自 URDF, damping_ratio_analysis.md)

# 默认参数: kpkd.py _TYPE_TABLE 原始值 + env_cfg viscous_friction + URDF I_link
JOINT_PARAMS: list[JointParams] = [
    #                name              Ia          kp     kd     d     I_link
    JointParams("HIP_PITCH",      0.2427264,  240.0,  5.80,  1.6,  3.6479),
    JointParams("HIP_ROLL",       0.14110848, 200.0,  4.20,  1.6,  4.4210),
    JointParams("HIP_YAW",        0.0448737,  113.0,  3.20,  1.0,  0.0409),
    JointParams("KNEE_PITCH",     0.2427264,  240.0,  5.80,  1.6,  2.0056),
    JointParams("ANKLE_PITCH",    0.0354625,   90.0,  0.45,  0.5,  0.4948),
    JointParams("ANKLE_ROLL",     0.0354625,   90.0,  0.45,  0.5,  0.0060),
    JointParams("TORSO_YAW",      0.0448737,  113.0,  3.20,  1.0,  1.0123),
    JointParams("SHOULDER_PITCH", 0.0354625,   90.0,  0.40,  0.5,  0.3811),
    JointParams("SHOULDER_ROLL",  0.0354625,   90.0,  0.40,  0.5,  0.6852),
    JointParams("SHOULDER_YAW",   0.0354625,   90.0,  0.40,  0.5,  0.0086),
    JointParams("ELBOW_PITCH",    0.0354625,   90.0,  0.40,  0.5,  0.3811),
    JointParams("ELBOW_YAW",      0.00671625,  50.0,  0.30,  0.1,  0.0036),
    JointParams("WRIST_PITCH",    0.005,       12.5,  0.13,  0.1,  0.0504),
    JointParams("WRIST_ROLL",     0.005,       12.5,  0.13,  0.1,  0.0098),
]

# ═══════════════════════════════════════════════════════════════════════════
# 计算函数
# ═══════════════════════════════════════════════════════════════════════════

def calc_omega_n(kp: float, I: float) -> float:
    """计算固有频率 [Hz].

    omega_n = sqrt(kp / I) / (2*pi)
    """
    return math.sqrt(kp / I) / (2.0 * math.pi)

def calc_zeta(kp: float, kd: float, d: float, I: float) -> float:
    """计算阻尼比.

    zeta = (d + kd) / (2 * sqrt(kp * I))
    """
    return (d + kd) / (2.0 * math.sqrt(kp * I))

def calc_all(params: list[JointParams], use_ilink: bool = True) -> list[dict]:
    """计算所有关节的 omega_n 和 zeta.

    Args:
        params: 关节参数列表
        use_ilink: True 用 I_total=Ia+I_link, False 只用 Ia

    Returns:
        结果字典列表
    """
    results = []
    for p in params:
        I_total = p.Ia + p.I_link
        I = I_total if use_ilink else p.Ia
        wn = calc_omega_n(p.kp, I)
        zeta = calc_zeta(p.kp, p.kd, p.d, I)
        results.append({
            "name": p.name,
            "Ia": p.Ia,
            "I_link": p.I_link,
            "I_total": I_total,
            "I_link_ratio": p.I_link / p.Ia,
            "kp": p.kp,
            "kd": p.kd,
            "d": p.d,
            "omega_n": wn,
            "zeta": zeta,
        })
    return results

# ═══════════════════════════════════════════════════════════════════════════
# 输出格式
# ═══════════════════════════════════════════════════════════════════════════

def print_table(results: list[dict], title: str, use_ilink: bool):
    """打印表格."""
    I_label = "I_total" if use_ilink else "Ia"
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"  I = {I_label}" + (" (= Ia + I_link, PhysX 实际值)" if use_ilink else " (纯 armature, 设计值)"))
    print(f"{'=' * 100}")
    header = f"{'Joint':<20} {'Ia':>8} {'I_link':>8} {'I_total':>8} {'I_link/Ia':>10} {'kp':>7} {'kd':>6} {'d':>5} {'omega_n':>9} {'zeta':>7}"
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['Ia']:>8.4f} "
            f"{r['I_link']:>8.4f} "
            f"{r['I_total']:>8.4f} "
            f"{r['I_link_ratio']:>9.1f}x "
            f"{r['kp']:>7.1f} "
            f"{r['kd']:>6.2f} "
            f"{r['d']:>5.1f} "
            f"{r['omega_n']:>8.2f}Hz "
            f"{r['zeta']:>7.3f}"
        )

    # 统计
    wns = [r["omega_n"] for r in results]
    zetas = [r["zeta"] for r in results]
    print("-" * 100)
    print(
        f"{'范围':<20} {'':>8} {'':>8} {'':>8} {'':>10} {'':>7} {'':>6} {'':>5} "
        f"{min(wns):>5.2f}-{max(wns):.2f}Hz {min(zetas):.3f}-{max(zetas):.3f}"
    )
    print(f"{'极差':<20} {'':>8} {'':>8} {'':>8} {'':>10} {'':>7} {'':>6} {'':>5} {max(wns)-min(wns):>8.2f}Hz {max(zetas)-min(zetas):>7.3f}")
    crit = sum(1 for z in zetas if z >= 0.9)
    under = sum(1 for z in zetas if z < 0.3)
    print(f"\n  zeta >= 0.9 (近临界阻尼): {crit}/{len(results)}")
    print(f"  zeta <  0.3 (欠阻尼):     {under}/{len(results)}")

def print_comparison(results_ia: list[dict], results_itot: list[dict]):
    """对比打印 Ia 和 I_total."""
    print(f"\n{'=' * 130}")
    print(f"  对比: 纯 Ia (设计值) vs I_total (PhysX 实际值)")
    print(f"{'=' * 130}")
    header = (
        f"{'Joint':<20} {'Ia':>8} {'I_link':>8} {'I_link/Ia':>10} "
        f"{'wn(Ia)':>8} {'zeta(Ia)':>9} "
        f"{'wn(Itot)':>9} {'zeta(Itot)':>11} "
        f"{'wn降幅':>7} {'zeta降幅':>8}"
    )
    print(header)
    print("-" * 130)
    for r_ia, r_it in zip(results_ia, results_itot):
        wn_drop = (1.0 - r_it["omega_n"] / r_ia["omega_n"]) * 100.0
        z_drop = (1.0 - r_it["zeta"] / r_ia["zeta"]) * 100.0
        print(
            f"{r_ia['name']:<20} "
            f"{r_ia['Ia']:>8.4f} "
            f"{r_ia['I_link']:>8.4f} "
            f"{r_ia['I_link_ratio']:>9.1f}x "
            f"{r_ia['omega_n']:>7.2f}Hz "
            f"{r_ia['zeta']:>9.3f} "
            f"{r_it['omega_n']:>8.2f}Hz "
            f"{r_it['zeta']:>11.3f} "
            f"{wn_drop:>6.0f}% "
            f"{z_drop:>7.0f}%"
        )

    # 统计
    wns_ia = [r["omega_n"] for r in results_ia]
    wns_it = [r["omega_n"] for r in results_itot]
    zs_ia = [r["zeta"] for r in results_ia]
    zs_it = [r["zeta"] for r in results_itot]
    print("-" * 130)
    print(f"\n  wn(Ia) 范围:     {min(wns_ia):.2f}-{max(wns_ia):.2f} Hz  (极差 {max(wns_ia)-min(wns_ia):.2f})")
    print(f"  wn(I_total) 范围: {min(wns_it):.2f}-{max(wns_it):.2f} Hz  (极差 {max(wns_it)-min(wns_it):.2f})")
    print(f"  zeta(Ia) 范围:     {min(zs_ia):.3f}-{max(zs_ia):.3f}  (极差 {max(zs_ia)-min(zs_ia):.3f})")
    print(f"  zeta(I_total) 范围: {min(zs_it):.3f}-{max(zs_it):.3f}  (极差 {max(zs_it)-min(zs_it):.3f})")
    print(f"  zeta >= 0.9:  Ia={sum(1 for z in zs_ia if z >= 0.9)}/{len(zs_ia)},  I_total={sum(1 for z in zs_it if z >= 0.9)}/{len(zs_it)}")
    print(f"  zeta <  0.3:  Ia={sum(1 for z in zs_ia if z < 0.3)}/{len(zs_ia)},  I_total={sum(1 for z in zs_it if z < 0.3)}/{len(zs_it)}")

def print_json(results: list[dict], use_ilink: bool):
    """JSON 输出."""
    output = {
        "inertia_mode": "I_total" if use_ilink else "Ia",
        "joints": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

def print_csv(results: list[dict], use_ilink: bool):
    """CSV 输出."""
    I_col = "I_total" if use_ilink else "Ia"
    print("joint,Ia,I_link,I_total,I_link_ratio,kp,kd,d,omega_n_hz,zeta")
    for r in results:
        print(
            f"{r['name']},{r['Ia']:.6f},{r['I_link']:.6f},{r['I_total']:.6f},"
            f"{r['I_link_ratio']:.2f},{r['kp']:.2f},{r['kd']:.4f},{r['d']:.2f},"
            f"{r['omega_n']:.4f},{r['zeta']:.6f}"
        )

# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="S800 关节固有频率与阻尼比计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
公式:
    omega_n = sqrt(kp / I) / (2*pi)   [Hz]
    zeta    = (d + kd) / (2 * sqrt(kp * I))

    I = Ia           (纯 armature, 设计值)
    I = Ia + I_link  (含连杆, PhysX 实际值)

参数来源:
    Ia, kp, kd  -> kpkd.py _TYPE_TABLE
    d           -> s800_pace_env_cfg.py viscous_friction
    I_link      -> URDF 下游连杆等效惯量
        """,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["ia", "itotal", "both"],
        default="ia",
        help="计算模式: ia=纯armature, itotal=含连杆, both=对比 (默认: both)",
    )
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--csv", action="store_true", help="CSV 格式输出")
    args = parser.parse_args()

    # JSON / CSV 模式只输出一种 I
    if args.json:
        use_ilink = args.mode != "ia"
        results = calc_all(JOINT_PARAMS, use_ilink=use_ilink)
        print_json(results, use_ilink)
        return

    if args.csv:
        use_ilink = args.mode != "ia"
        results = calc_all(JOINT_PARAMS, use_ilink=use_ilink)
        print_csv(results, use_ilink)
        return

    # 表格输出
    if args.mode == "ia":
        results = calc_all(JOINT_PARAMS, use_ilink=False)
        print_table(results, "固有频率与阻尼比 (纯 Ia)", use_ilink=False)

    elif args.mode == "itotal":
        results = calc_all(JOINT_PARAMS, use_ilink=True)
        print_table(results, "固有频率与阻尼比 (I_total = Ia + I_link)", use_ilink=True)

    else:  # both
        results_ia = calc_all(JOINT_PARAMS, use_ilink=False)
        results_itot = calc_all(JOINT_PARAMS, use_ilink=True)
        print_comparison(results_ia, results_itot)

if __name__ == "__main__":
    main()
