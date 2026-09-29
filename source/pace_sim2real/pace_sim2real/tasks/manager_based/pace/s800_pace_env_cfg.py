# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

# config for s800

from isaaclab.utils import configclass

# from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
import isaaclab.sim as sim_utils

S800_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*"],
    saturation_effort=415.0, # 电机峰值扭矩 from XML
    # effort_limit=415.0, # 电机持续力矩 from XML
    # velocity_limit=20.0, # 电机最大速度 保守估计
    # stiffness={".*": 100.0},  # P gain in Nm/rad 
    # damping={".*": 2.0},  # D gain in Nm s/rad
    effort_limit={
        ".*HIP_PITCH.*": 415.0, ".*HIP_ROLL.*": 370.0, ".*HIP_YAW.*": 222.0,
        ".*KNEE_PITCH.*": 415.0, ".*ANKLE.*": 160.0, ".*TORSO.*": 222.0,
        ".*SHOULDER.*": 160.0, ".*ELBOW_PITCH.*": 160.0, ".*ELBOW_YAW.*": 52.0,
        ".*WRIST.*": 20.0,
    },

    velocity_limit={
        ".*HIP_PITCH.*": 25.96, ".*HIP_ROLL.*": 25.31, ".*HIP_YAW.*": 23.19,
        ".*KNEE_PITCH.*": 25.96, ".*ANKLE.*": 33.51, ".*TORSO.*": 23.19,
        ".*SHOULDER.*": 33.51, ".*ELBOW_PITCH.*": 33.51, ".*ELBOW_YAW.*": 35.2,
        ".*WRIST.*": 150.0,
    },

    # ── 方案 C（含延时稳定性修正）────────────────────────────────────────────
    # 原文依据 (PACE p.7)：「Increasing the gains shifts the dominant poles to higher
    # frequencies... empirically, higher gains did not improve identification quality but
    # instead pushed the closed-loop poles beyond the feasible excitation bandwidth.」
    # 即闭环极点 ω_cl = sqrt(kp/I_total) 要落在 chirp 带内，否则关节全程紧跟参考、
    # 跟踪误差 e ≈ J·q̈_des/kp 过小 → armature 不可辨识。
    #
    # ⚠️ 但还有第二个约束：环路延时。指令延时 5 步 @400Hz = 12.5ms，贡献相位滞后 ω_c·T。
    #    PM = atan(kd·ω_c/kp) - ω_c·T  ⇒  在 T=12.5ms 下 ω_c 硬上限约 7.8Hz。
    #    直接照搬 kpkd 的 kp 会让 5 个轻惯量关节越界 —— ELBOW_YAW 的 PM=-12°，
    #    实测无预警地起振，振荡频率 12.5Hz ≈ 预测的增益穿越频率 12.3Hz。
    #    这 5 个的 kp 下调到 PM=30°（其余 9 类保留 kpkd 原值）。
    #    注：这两条约束同向 —— 降 kp 同时把极点拉回 chirp 带内。
    stiffness={
        ".*HIP_PITCH.*": 240.0, ".*HIP_ROLL.*": 200.0, ".*HIP_YAW.*": 35.11,
        ".*KNEE_PITCH.*": 240.0, ".*ANKLE_PITCH.*": 90.0, ".*ANKLE_ROLL.*": 17.16, ".*TORSO.*": 113.0,
        ".*SHOULDER_PITCH.*": 90.0, ".*SHOULDER_ROLL.*": 90.0, ".*SHOULDER_YAW.*": 17.89,
        ".*ELBOW_PITCH.*": 90.0, ".*ELBOW_YAW.*": 3.99,
        ".*WRIST_PITCH.*": 12.5, ".*WRIST_ROLL.*": 5.17,
    },

    # stiffness={
    #     ".*HIP_PITCH.*": 240.00, ".*HIP_ROLL.*": 200.00, ".*HIP_YAW.*": 80.0,
    #     ".*KNEE_PITCH.*": 240.0, ".*ANKLE_PITCH.*": 50.0, ".*ANKLE_ROLL.*": 50.0, ".*TORSO.*": 80.0,
    #     ".*SHOULDER_PITCH.*": 60.0, ".*SHOULDER_ROLL.*": 60.0, ".*SHOULDER_YAW.*": 60.0,
    #     ".*ELBOW_PITCH.*": 60.0, ".*ELBOW_YAW.*": 10.0,
    #     ".*WRIST.*": 8.0,
    # },

    # kd = 2ζ·sqrt(kp·I_total) − d，取 ζ = 0.40，kp 用上面调整后的值。
    # 注：上一版曾据「ω_cl 只由 kp 决定、ζ 主要由 kd 决定」把两者独立设定 —— 那个说法
    # 只在无延时模型里成立。加上 12.5ms 延时后两者通过相位裕度耦合，必须先按稳定性
    # 定下 kp，再算 kd。
    damping={
        ".*HIP_PITCH.*": 22.8458, ".*HIP_ROLL.*": 22.565, ".*HIP_YAW.*": 0.3885,
        ".*KNEE_PITCH.*": 16.9833, ".*ANKLE_PITCH.*": 5.0268, ".*ANKLE_ROLL.*": 0.175, ".*TORSO.*": 7.744,
        ".*SHOULDER_PITCH.*": 4.3986, ".*SHOULDER_ROLL.*": 5.943, ".*SHOULDER_YAW.*": 0.2107,
        ".*ELBOW_PITCH.*": 4.3986, ".*ELBOW_YAW.*": 0.0621,
        ".*WRIST_PITCH.*": 0.5657, ".*WRIST_ROLL.*": 0.1214,
    },

    # damping={
    #     ".*HIP_PITCH.*": 5.8, ".*HIP_ROLL.*": 4.2, ".*HIP_YAW.*": 2.0,
    #     ".*KNEE_PITCH.*": 5.8, ".*ANKLE_PITCH.*": 0.6, ".*ANKLE_ROLL.*": 0.6, ".*TORSO.*": 2.0,
    #     ".*SHOULDER_PITCH.*": 0.6, ".*SHOULDER_ROLL.*": 0.6, ".*SHOULDER_YAW.*": 0.6,
    #     ".*ELBOW_PITCH.*": 0.6, ".*ELBOW_YAW.*": 0.25,
    #     ".*WRIST_PITCH.*": 0.13, ".*WRIST_ROLL.*": 0.13,
    # },

    encoder_bias={".*": 0.0},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={".*": 2.0},  # static friction coefficient (Nm) from XML
    dynamic_friction={".*": 2.0},  # dynamic friction coefficient (Nm) 估计
    viscous_friction={
        # Per motor type: large (7520-22) 1.6, medium (7520-14) 1.0,
        # small (5020) 0.5, smallest (4010) 0.1
        ".*HIP_PITCH.*": 1.6, ".*HIP_ROLL.*": 1.6, ".*KNEE_PITCH.*": 1.6,
        ".*HIP_YAW.*": 1.0, ".*TORSO.*": 1.0,
        ".*ANKLE.*": 0.5, ".*SHOULDER.*": 0.5, ".*ELBOW_PITCH.*": 0.5,
        ".*ELBOW_YAW.*": 0.1, ".*WRIST.*": 0.1,
    },
    max_delay=10,  # max delay in simulation steps
)

@configclass
class S800PaceCfg(PaceCfg):
    """Pace configuration for s800 robot."""
    robot_name: str = "s800_sim"
    data_dir: str = "s800_sim/chirp_data.pt"  # located in pace_sim2real/data/s800/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((109, 2))  # s800 has 27 dof, 4n+1 -> 4 * 27 + 1 = 109 parameters to optimize
    joint_order: list[str] = [
        "J00_HIP_PITCH_L",
        "J01_HIP_ROLL_L",
        "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L",
        "J04_ANKLE_PITCH_L",
        "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R",
        "J07_HIP_ROLL_R",
        "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R",
        "J10_ANKLE_PITCH_R",
        "J11_ANKLE_ROLL_R",
        "J12_TORSO_YAW",
        "J13_SHOULDER_PITCH_L",
        "J14_SHOULDER_ROLL_L",
        "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L",
        "J17_ELBOW_YAW_L",
        "J18_WRIST_PITCH_L",
        "J19_WRIST_ROLL_L",
        "J27_SHOULDER_PITCH_R",
        "J28_SHOULDER_ROLL_R",
        "J29_SHOULDER_YAW_R",
        "J30_ELBOW_PITCH_R",
        "J31_ELBOW_YAW_R",
        "J32_WRIST_PITCH_R",
        "J33_WRIST_ROLL_R"
    ]

    def __post_init__(self): #TODO
        # set bounds for parameters
        # self.bounds_params[:27, 0] = 1e-4
        # self.bounds_params[:27, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]

        # --- Armature [0:27]: per-joint-type bounds, bracket the round-trip GT ---
        # GT is the 4-group ground truth injected by data_collection.py (see there).
        # Bounds must straddle GT, otherwise the optimizer can never reach the true
        # value and the round-trip test fails for a trivial reason.
        armature_bounds = {
            "HIP_PITCH":      (0.05, 0.50),   # GT=0.24
            "HIP_ROLL":       (0.03, 0.30),   # GT=0.14
            "HIP_YAW":        (0.01, 0.15),   # GT=0.05
            "KNEE_PITCH":     (0.05, 0.50),   # GT=0.24
            "ANKLE_PITCH":    (0.005, 0.15),  # GT=0.05
            "ANKLE_ROLL":     (0.005, 0.15),  # GT=0.05
            "TORSO_YAW":      (0.01, 0.15),   # GT=0.05
            "SHOULDER_PITCH": (0.005, 0.15),  # GT=0.05
            "SHOULDER_ROLL":  (0.005, 0.15),  # GT=0.05
            "SHOULDER_YAW":   (0.005, 0.15),  # GT=0.05
            "ELBOW_PITCH":    (0.005, 0.15),  # GT=0.05
            "ELBOW_YAW":      (0.001, 0.03),  # GT=0.008
            "WRIST_PITCH":    (0.0005, 0.02), # GT=0.008
            "WRIST_ROLL":     (0.0005, 0.02), # GT=0.008
        }

        damping_bounds = {
            "HIP_PITCH":      (0.0, 3.0),   # GT=1.6
            "HIP_ROLL":       (0.0, 3.0),   # GT=1.6
            "HIP_YAW":        (0.0, 2.0),   # GT=1
            "KNEE_PITCH":     (0.0, 3.0),   # GT=1.6
            "ANKLE_PITCH":    (0.0, 1.0),  # GT=0.5
            "ANKLE_ROLL":     (0.0, 1.0),  # GT=0.5
            "TORSO_YAW":      (0.0, 2.0),   # GT=1.0
            "SHOULDER_PITCH": (0.0, 1.0),  # GT=0.5
            "SHOULDER_ROLL":  (0.0, 1.0),  # GT=0.5
            "SHOULDER_YAW":   (0.0, 1.0),  # GT=0.5
            "ELBOW_PITCH":    (0.0, 1.0),  # GT=0.5
            "ELBOW_YAW":      (0.0, 0.5),  # GT=0.1
            "WRIST_PITCH":    (0.0, 0.5), # GT=0.1
            "WRIST_ROLL":     (0.0, 0.5), # GT=0.1
        }

        print(f"joint order = {self.joint_order}")

        for i, name in enumerate(self.joint_order):
            jtype = name[4:].rsplit("_", 1)[0] if name.endswith(("_L", "_R")) else name[4:]  # "J00_HIP_PITCH_L" -> "HIP_PITCH", "J12_TORSO_YAW" -> "TORSO_YAW"
            # armature [0:27]
            lo, hi = armature_bounds[jtype]
            self.bounds_params[i, 0] = lo
            self.bounds_params[i, 1] = hi
            # damping (viscous friction) [27:54]
            d_lo, d_hi = damping_bounds[jtype]
            self.bounds_params[27 + i, 0] = d_lo
            self.bounds_params[27 + i, 1] = d_hi

        # friction [54:81] — 上限必须覆盖 data_collection.py 注入的真值(最大 1.0)。
        # 曾设 0.6 而 GT=1.5：搜索空间不包含真值，round-trip 必然失败。
        self.bounds_params[54:81, 0] = 0.0
        self.bounds_params[54:81, 1] = 2.0  # friction between 0.0 - 2.0 [Nm]

        self.bounds_params[81:108, 0] = -0.1
        self.bounds_params[81:108, 1] = 0.1  # bias between -0.1 - 0.1 [rad]

        self.bounds_params[108, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]

# @configclass
# class S800PaceSceneCfg(PaceSim2realSceneCfg):
#     """Configuration for s800 robot in Pace Sim2Real environment."""
#     robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
#                                                   actuators={"legs": ANYDRIVE_PACE_ACTUATOR_CFG})

@configclass
class S800PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for s800 robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path="/home/samsung/Humanoid/pace-sim2real/assets_robot/engineai_S800/usd/serial_s800.usda",
            # usd_path="/home/samsung/Humanoid/pace-sim2real/assets_robot/s800/usd_merge_joints/serial_robot_s_42dof/serial_robot_s_42dof.usda",
            # usd_path="/home/samsung/Humanoid/pace-sim2real/assets_robot/s800/usd_no_inertia/serial_robot_s_42dof/serial_robot_s_42dof.usda",
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)), # 需要将s800悬空 避免脚接触地面
        actuators={"joints": S800_ACTUATOR_CFG},
    )

@configclass
class S800PaceEnvCfg(PaceSim2realEnvCfg):

    scene: S800PaceSceneCfg = S800PaceSceneCfg()
    sim2real: PaceCfg = S800PaceCfg()

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
