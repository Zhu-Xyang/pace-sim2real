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

    stiffness={
        # PD gains adjusted using I_total = Ia + I_link (from URDF downstream links):
        #   ω_n = sqrt(kp / I_total),  ζ = (d + kd) / (2 * sqrt(kp * I_total))
        # Target: ω_n ∈ [2, 5] Hz, ζ ∈ [0.3, 0.7] (minimal changes from original).
        # Original design used Ia only; PhysX uses I_total, causing ζ to drop.
        ".*HIP_PITCH.*": 632.95, ".*HIP_ROLL.*": 742.19, ".*HIP_YAW.*": 83.67,
        ".*KNEE_PITCH.*": 365.77, ".*ANKLE_PITCH.*": 86.27, ".*ANKLE_ROLL.*": 40.47, ".*TORSO.*": 171.99,
        ".*SHOULDER_PITCH.*": 68.6, ".*SHOULDER_ROLL.*": 117.25, ".*SHOULDER_YAW.*": 43.0,
        ".*ELBOW_PITCH.*": 68.6, ".*ELBOW_YAW.*": 10.04,
        ".*WRIST.*": 12.5,
    },

    # stiffness={
    #     ".*HIP_PITCH.*": 240.00, ".*HIP_ROLL.*": 200.00, ".*HIP_YAW.*": 80.0,
    #     ".*KNEE_PITCH.*": 240.0, ".*ANKLE_PITCH.*": 50.0, ".*ANKLE_ROLL.*": 50.0, ".*TORSO.*": 80.0,
    #     ".*SHOULDER_PITCH.*": 60.0, ".*SHOULDER_ROLL.*": 60.0, ".*SHOULDER_YAW.*": 60.0,
    #     ".*ELBOW_PITCH.*": 60.0, ".*ELBOW_YAW.*": 10.0,
    #     ".*WRIST.*": 8.0,
    # },

    damping={
        ".*HIP_PITCH.*": 30.16, ".*HIP_ROLL.*": 35.64, ".*HIP_YAW.*": 2.64,
        ".*KNEE_PITCH.*": 16.75, ".*ANKLE_PITCH.*": 3.83, ".*ANKLE_ROLL.*": 1.26, ".*TORSO.*": 7.63,
        ".*SHOULDER_PITCH.*": 2.92, ".*SHOULDER_ROLL.*": 5.38, ".*SHOULDER_YAW.*": 1.37,
        ".*ELBOW_PITCH.*": 2.92, ".*ELBOW_YAW.*": 0.34,
        ".*WRIST_PITCH.*": 0.43, ".*WRIST_ROLL.*": 0.18,
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

        # --- Armature [0:27]: per-joint-type bounds centered on kpkd.py values ---
        # Format: (lower, upper) for each joint type
        armature_bounds = {
            "HIP_PITCH":      (0.05, 0.50),   # GT=0.2427
            "HIP_ROLL":       (0.03, 0.30),   # GT=0.1411
            "HIP_YAW":        (0.01, 0.15),   # GT=0.0449
            "KNEE_PITCH":     (0.05, 0.50),   # GT=0.2427
            "ANKLE_PITCH":    (0.005, 0.15),  # GT=0.0355
            "ANKLE_ROLL":     (0.005, 0.15),  # GT=0.0355
            "TORSO_YAW":      (0.01, 0.15),   # GT=0.0449
            "SHOULDER_PITCH": (0.005, 0.15),  # GT=0.0355
            "SHOULDER_ROLL":  (0.005, 0.15),  # GT=0.0355
            "SHOULDER_YAW":   (0.005, 0.15),  # GT=0.0355
            "ELBOW_PITCH":    (0.005, 0.15),  # GT=0.0355
            "ELBOW_YAW":      (0.001, 0.03),  # GT=0.0067
            "WRIST_PITCH":    (0.0005, 0.02), # GT=0.005
            "WRIST_ROLL":     (0.0005, 0.02), # GT=0.005
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

        self.bounds_params[54:81, 0] = 0.0
        self.bounds_params[54:81, 1] = 0.6  # friction between 0.0 - 0.5

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
            # usd_path="/home/samsung/Humanoid/pace-sim2real/assets_robot/s800/usd/serial_robot_s_42dof/serial_robot_s_42dof.usda",
            usd_path="/home/samsung/Humanoid/pace-sim2real/assets_robot/s800/usd_merge_joints/serial_robot_s_42dof/serial_robot_s_42dof.usda",
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

        # URDF 转换时已通过 --fix-base 固定基座，不需要 Isaac Lab 再创建固定关节
        self.scene.robot.spawn.articulation_props.fix_root_link = False

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
