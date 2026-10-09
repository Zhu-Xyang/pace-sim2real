# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

# config for g1 (Unitree G1, 29 DOF, g1_29dof_rev_1_0)

from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch
from isaaclab.sim.spawners.from_files import UsdFileCfg
import isaaclab.sim as sim_utils

# ── 资产 ────────────────────────────────────────────────────────────────────
# 由 assets_robot/g1_description/g1_29dof_rev_1_0.urdf 转换而来，转换设置与 S800 一致：
#   scripts/tools/convert_urdf.py <urdf> <usda> --fix-base --merge-joints
# 已核验（pxr 直接读 USD，与 URDF 逐项对比）：
#   · 29 个 revolute 关节名字全对，root_joint 挂 ArticulationRootAPI（--fix-base 造的）
#   · 9 个 fixed 关节合并 7 个；两只橡胶手带 dont_collapse="true" 故保留为独立刚体（无害，
#     它们不是 DOF，不进 joint_names）
#   · 总质量 33.3411 kg 完全一致；逐刚体质量/质心/主惯量一致（ΔI<3e-8 kg·m²）
#   · ⚠️ URDF 里 world→pelvis 的 floating 关节整段被 <!-- --> 注释掉了，真实 root 就是
#     pelvis（没有 world 层），和 S800 一样，所以 fix_root_link 的语义与 S800 相同
#   · ⚠️ USD 的关节限位单位是「度」（UsdPhysics 的通用约定，S800 也一样），不用处理
G1_USD_PATH = "/home/samsung/Humanoid/pace-sim2real/assets_robot/g1_description/usd/g1_29dof_rev_1_0.usda"

# ── 29 个关节的顺序（4 段内部都按这个顺序）────────────────────────────────
# 顺序本身是约定，不是 PhysX 的 DOF 顺序：代码全部按名字查
# articulation.joint_names.index(name)，只要 data_collection 的 GT 数组与之对齐即可。
G1_JOINT_ORDER = [
    # 左腿
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # 右腿
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰（G1 有 3 个，S800 只有 1 个）
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # 右臂
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ── armature：两个不同的角色，别混 ─────────────────────────────────────────
# ① G1_ARMATURE_GT = XML (assets_robot/g1/g1_29dof.xml) 给的官方值，全场 0.01。
#    **注入仿真的 round-trip 真值、以及 armature bounds 的中心都用它** —— 官方来源、可追溯。
# ② G1_MOTOR_ARMATURE = 按电机档的反射转子惯量（转子惯量×减速比²）。
#    只用来**定 kp/kd 的初始口径**（kp = Ia·ω²，见下面）：增益是控制设计量，XML 里没有
#    （那边用 <motor> 力矩控制），而不同关节要拿到相同带宽就需要不同的 kp，所以这里必须
#    按各关节实际的反射惯量来算。
G1_ARMATURE_GT = 0.01

# kp/kd 口径用的按档反射转子惯量
G1_MOTOR_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
# 官方增益口径：kp = armature·(2π·10Hz)²，kd = 2·2.0·armature·(2π·10Hz)
_G1_W = 10 * 2.0 * 3.1415926535
_G1_ZETA = 2.0


def _pk(motor):
    """(kp, kd) 按 BeyondMimic 的官方口径。"""
    ia = G1_MOTOR_ARMATURE[motor]
    return ia * _G1_W**2, 2.0 * _G1_ZETA * ia * _G1_W


# 每个关节类型: (电机档, 电机数, effort[Nm], velocity[rad/s], 粘性摩擦标称)
# effort/velocity 取自 BeyondMimic 的 G1 配置，与 URDF 的 <limit> 一致（ankle/waist_roll/pitch
# 例外：URDF 35/30，BM 50/37 —— 因为那两个关节是双电机，BM 按 2× 给）。
# 粘性摩擦取自 assets_robot/g1/g1_29dof.xml 的 <default>（每类都给了 damping=0.05）。
# ⚠️ 早先这里照搬过 S800 同族电机的标称（1.6/1.0/0.5/0.1）—— 那是 S800 的量纲，G1 没有
#    独立的真机标定值，用官方 XML 的值比跨机器人借用可追溯。真机标定后再替换。
G1_TYPE_TABLE = {
    #                  电机        个数  effort  vel   粘性
    "HIP_PITCH":      ("7520_14",   1,   88.0,  32.0,  0.05),
    "HIP_ROLL":       ("7520_22",   1,  139.0,  20.0,  0.05),
    "HIP_YAW":        ("7520_14",   1,   88.0,  32.0,  0.05),
    "KNEE":           ("7520_22",   1,  139.0,  20.0,  0.05),
    "ANKLE_PITCH":    ("5020",      2,   50.0,  37.0,  0.05),
    "ANKLE_ROLL":     ("5020",      2,   50.0,  37.0,  0.05),
    "WAIST_YAW":      ("7520_14",   1,   88.0,  32.0,  0.05),
    "WAIST_ROLL":     ("5020",      2,   50.0,  37.0,  0.05),
    "WAIST_PITCH":    ("5020",      2,   50.0,  37.0,  0.05),
    "SHOULDER_PITCH": ("5020",      1,   25.0,  37.0,  0.05),
    "SHOULDER_ROLL":  ("5020",      1,   25.0,  37.0,  0.05),
    "SHOULDER_YAW":   ("5020",      1,   25.0,  37.0,  0.05),
    "ELBOW":          ("5020",      1,   25.0,  37.0,  0.05),
    "WRIST_ROLL":     ("5020",      1,   25.0,  37.0,  0.05),
    "WRIST_PITCH":    ("4010",      1,    5.0,  22.0,  0.05),
    "WRIST_YAW":      ("4010",      1,    5.0,  22.0,  0.05),
}
# 库仑摩擦上界：**不能用「峰值力矩的固定百分比」**。XML 的 frictionloss 折算到峰值力矩是
# 0.14%（hip_roll 0.2/139）~2%（腕 0.1/5），差 14 倍 —— 任何单一百分比都会让一部分关节
# 的真值贴在 0 那面墙上（真值离下界太近 = CMA-ES 分辨率不够，S800 的 uniform 实验就是这么
# 栽的），或者干脆越界（上界 1% 时腕的 0.1 > 0.05）。
# 改为锚在 XML 的官方值上 ×3：真值恒定落在区间 1/3 处，而且区间仍来自独立先验（官方
# datasheet 值 ± 不确定度），不是拿 GT 反推。真机标定后换掉这个 nominal 即可。
G1_FRICTION_NOMINAL = {"_default": 0.2, "WRIST_PITCH": 0.1, "WRIST_YAW": 0.1}
G1_FRICTION_BOUND_FACTOR = 3.0


def g1_joint_type(joint_name: str) -> str:
    """'left_hip_pitch_joint' -> 'HIP_PITCH'；'waist_yaw_joint' -> 'WAIST_YAW'"""
    s = joint_name.replace("left_", "").replace("right_", "")
    return s[: -len("_joint")].upper()


# 按上面两张表展开成逐关节的字典（键用完整关节名，Isaac Lab 的 joint_names_expr 支持正则，
# 关节名里除了下划线没有特殊字符，所以精确名直接当正则用）
_G1_TYPE_OF = {n: g1_joint_type(n) for n in G1_JOINT_ORDER}
G1_ARMATURE = {n: G1_MOTOR_ARMATURE[G1_TYPE_TABLE[t][0]] * G1_TYPE_TABLE[t][1]
               for n, t in _G1_TYPE_OF.items()}
G1_EFFORT_LIMIT = {n: G1_TYPE_TABLE[t][2] for n, t in _G1_TYPE_OF.items()}
G1_VELOCITY_LIMIT = {n: G1_TYPE_TABLE[t][3] for n, t in _G1_TYPE_OF.items()}
G1_VISCOUS = {n: G1_TYPE_TABLE[t][4] for n, t in _G1_TYPE_OF.items()}
G1_STIFFNESS = {n: _pk(G1_TYPE_TABLE[t][0])[0] for n, t in _G1_TYPE_OF.items()}
G1_DAMPING = {n: _pk(G1_TYPE_TABLE[t][0])[1] for n, t in _G1_TYPE_OF.items()}

# ── 执行器 ──────────────────────────────────────────────────────────────────
# ⚠️⚠️ kp/kd 目前是 BeyondMimic 的**官方值**。它们本身可用，但**必须过一遍相位裕度**：
#
# 官方口径 kp = armature·ω²（ω=10Hz）⇒ 闭环极点 ω_cl = 10Hz·√(Ia/I_total)。
# 用 URDF 算出的下游等效惯量 I_total（零位，含 armature）代入，得到：
#
#   类型            I_link   I_total   ω_cl     在 chirp 带(0.1-10Hz)内?
#   HIP_PITCH      0.9150   0.9251    1.05 Hz  ✓
#   HIP_ROLL       0.7735   0.7986    1.77 Hz  ✓
#   KNEE           0.1156   0.1407    4.22 Hz  ✓
#   HIP_YAW        0.0383   0.0485    4.58 Hz  ✓
#   ANKLE_PITCH    0.0028   0.0100    8.50 Hz  ✓（贴上限，带内余量仅 1.5Hz）
#   ANKLE_ROLL     0.0004   0.0076    9.74 Hz  ⚠️（几乎顶到 10Hz）
#   WAIST_YAW      0.2611   0.2713    1.94 Hz  ✓
#   WAIST_ROLL     0.6871   0.6944    1.02 Hz  ✓
#   WAIST_PITCH    0.6101   0.6173    1.08 Hz  ✓
#   SHOULDER_PITCH 0.1299   0.1335    1.64 Hz  ✓
#   SHOULDER_ROLL  0.0810   0.0847    2.07 Hz  ✓
#   SHOULDER_YAW   0.0412   0.0448    2.84 Hz  ✓
#   ELBOW          0.0345   0.0381    3.08 Hz  ✓
#   WRIST_ROLL     0.0004   0.0040    9.52 Hz  ⚠️（几乎顶到 10Hz）
#   WRIST_PITCH    0.0048   0.0091    6.85 Hz  ✓
#   WRIST_YAW      0.0018   0.0061    8.35 Hz  ✓（余量 1.65Hz）
#
# ⚠️ 这张表以前是拿 4Hz 扫频带算的，结论是「7 个类型极点出带 ⇒ armature 采不出来」。
#    2026-10-08 核对：G1 实际采的是 **0.1-10Hz**（从 des_dof_pos 反解瞬时频率确认），
#    所以**没有关节因出带而不可辨识**。PACE p.7 的那条约束当前是满足的。
#
# 真正的矛盾是**稳定性**：力矩延时在回路内（1+C·P·e^{-sT}=0），延时会直接吃相位裕度。
# ⚠️ 2026-10-09 更新：延时真值已从 5 步(12.5ms) 改到 **2 步(5ms)**（见 robot_tables.py）。
#    按 S800 文档反推的判据 ω_c·T ≲ 0.613 rad：踝 ω_cl=9.74Hz 在 5 步时滞后 43.8° 超限，
#    实测后果是腿部数据里 hip_yaw 54.6%、ankle_pitch 49.4% 的误差能量来自 18.9Hz 极限环，
#    损失地形被搅乱、拟合滑进补偿谷。改成 2 步后滞后 17.5°，全部关节进安全区。
#    这张表里 ω_cl 8.5~9.7Hz 的三个（ANKLE_PITCH/ANKLE_ROLL/WRIST_ROLL）因此**不再需要**
#    为了稳定性降增益 —— 它们仍在 3Hz 带外一点点，但那是可辨识性余量问题（sim2sim 无噪声
#    时可接受），不再是失稳问题。
#
# 重做的口径（沿用 S800 的方案，见 s800_pace_env_cfg.py 的"方案 C"注释）：
#   1) kp = I_total·(2πf)², kd = 2ζ√(kp·I_total) − d
#      S800 取值：连杆主导（I_link/Ia ≳ 8）→ f=2.03Hz ζ=0.320；
#                 电机主导（I_link/Ia ≲ 2）→ f=4.97Hz ζ=0.679
#      G1 的 I_link/Ia 见上表隐含（I_total/Ia − 1）
#   2) 相位裕度复核（12.5ms 延时下 PM = atan(kd·ω_c/kp) − ω_c·T ≥ 30°）。
#      S800 实测有 5 个轻惯量关节 PM=-12° 直接起振，必须逐个过一遍。
#      ⚠️ scripts/bode_phase_margin.py 的 PLANT 字典是 S800 的，需换 G1 参数表。
#   3) 幅度还要受行程、力矩权限、自碰撞三重限制（见 data_collection.py）
G1_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*"],
    # ⚠️ saturation_effort 是 DCMotor 模型的单标量，这里取全场最大 effort（139）。
    #    对 5Nm 的手腕来说这个力矩-转速曲线是失真的，但 S800 当时也是这么处理的
    #    （415 对全场）。如果要更准，可以拆成 legs/waist/arms 三个 drive 各给一个值。
    saturation_effort=139.0,
    effort_limit=G1_EFFORT_LIMIT,
    velocity_limit=G1_VELOCITY_LIMIT,
    stiffness=G1_STIFFNESS,
    damping=G1_DAMPING,
    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    friction={".*": 2.0},
    dynamic_friction={".*": 2.0},
    viscous_friction=G1_VISCOUS,
    # 延时缓冲的历史长度。DelayBuffer 允许的时间滞后是 0..max_delay+1 ——
    # bounds_params[116] 的上界必须落在这个范围内，否则采样越界会 ValueError。
    # G1 取 4（允许 0..5 步）：力矩延时在回路内，踝的 ω_cl=9.74Hz 在 5 步(12.5ms) 时
    # 相位滞后 43.8° 已超出 ω_c·T ≲ 0.613 rad 的稳定判据，实测会起 18.9Hz 极限环
    # （腿部数据 hip_yaw 54.6% / ankle_pitch 49.4% 的误差能量）。4 步(10ms)=35.1° 是边界。
    # ⚠️ 与 robot_tables.py 的 delay_gt 配套改。
    max_delay=4,
)


@configclass
class G1PaceCfg(PaceCfg):
    """Pace configuration for the Unitree G1 (29 DOF)."""
    robot_name: str = "g1_sim"
    data_dir: str = "g1_sim/chirp_data.pt"  # located in pace_sim2real/data/g1_sim/chirp_data.pt
    # 4n+1 -> 4 * 29 + 1 = 117 个待辨识参数
    bounds_params: torch.Tensor = torch.zeros((117, 2))
    joint_order: list[str] = list(G1_JOINT_ORDER)

    def __post_init__(self):
        # --- Armature [0:29]: 以 XML 的官方 armature 为中心，区间 [0.2×, 4×] ---
        # ⚠️ 和 S800 一样：真值必须落在区间内，否则 CMA-ES 会贴着边界收敛且不报错。
        #    中心用 XML 的值（0.01，全场统一）而不是按电机档，是为了和注入的 GT 同源 ——
        #    否则 0.01 会落在 "按电机档区间" 的 5%~68% 不同位置上，余量很不均匀。
        for i, name in enumerate(self.joint_order):
            self.bounds_params[i, 0] = 0.2 * G1_ARMATURE_GT
            self.bounds_params[i, 1] = 4.0 * G1_ARMATURE_GT
            # --- damping (viscous friction) [29:58]: 下界 0，上界 3× 标称 ---
            self.bounds_params[29 + i, 0] = 0.0
            self.bounds_params[29 + i, 1] = 3.0 * G1_VISCOUS[name]
            # --- friction (coulomb) [58:87]: 上界 = XML 官方值 × 3（见上面 G1_FRICTION_NOMINAL）---
            # 下界一律 0（摩擦可以接近 0）。⚠️ 摩擦须明显小于该关节能产生的 PD 力矩，
            # 否则关节根本不动、参数不可辨识（S800 用 [1.5]*27 把手腕锁死过）。
            self.bounds_params[58 + i, 0] = 0.0
            self.bounds_params[58 + i, 1] = (G1_FRICTION_BOUND_FACTOR
                                             * G1_FRICTION_NOMINAL.get(_G1_TYPE_OF[name],
                                                                       G1_FRICTION_NOMINAL["_default"]))

        # --- bias [87:116] / delay [116] ---
        self.bounds_params[87:116, 0] = -0.1
        self.bounds_params[87:116, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        # delay [0, 4] 步：真值 2 落在正中；上限 4 步(10ms) 正好是踝(ω_cl=9.74Hz)的稳定边界，
        # 保证整个搜索种群都在不失稳的一侧（见上面 max_delay 的注释）。
        self.bounds_params[116, 0] = 0.0
        self.bounds_params[116, 1] = 4.0  # delay between 0.0 - 4.0 [sim steps]

        print(f"joint order = {self.joint_order}")


@configclass
class G1PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for the Unitree G1 in Pace Sim2Real environment."""
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=G1_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # ⚠️ 自碰撞开着。G1 的手臂在零位贴身体，腰 roll/pitch 行程只有 ±0.52 rad，
                #    幅度给大了会自撞；一旦发生接触，腿和臂就通过接触力耦合，分组优化的
                #    解耦前提就破了（S800 吃过手腕撞髋的亏）。
                enabled_self_collisions=True,
                # 8/4 与 BeyondMimic 的 G1 配置一致（S800 用的是 4/0）。
                # 影响的是约束求解精度，data_collection 与 fit 共用本 cfg 所以会互相抵消，
                # 但真机对照时更准的求解更好。
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        # 需要把 G1 悬空，避免脚接触地面（和 S800 同样处理）。G1 站立高度约 0.79m。
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
        actuators={"joints": G1_ACTUATOR_CFG},
    )


@configclass
class G1PaceEnvCfg(PaceSim2realEnvCfg):

    scene: G1PaceSceneCfg = G1PaceSceneCfg()
    sim2real: PaceCfg = G1PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 必须为 True。URDF 导入时 --fix-base 造了一个 root_joint（固定关节，同时挂着
        # ArticulationRootAPI），fix_root_link 的作用是「开关这个已存在的关节」而不是
        # 「再创建一个」：False 会把 articulation root 一起关掉，PhysX 报
        # "did not match any rigid bodies"。（基类已设 True，这里显式写出以免被误改。）
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
