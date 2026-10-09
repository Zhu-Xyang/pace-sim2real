# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

# import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##

@configclass
class PaceSim2realSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    env_spacing: float = 2.5
    num_envs: int = 4096

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)  # actions = absolute joint position targets

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=False)

@configclass
class CMAESOptimizerCfg:
    """CMA-ES optimizer configuration."""
    max_iteration: int = 15000 # 从四足12dof到s800 27dof 增加迭代次数
    epsilon: float = 1e-2
    sigma: float = 0.5
    save_interval: int = 25
    save_optimization_process: bool = False  # consume more disk space if True, saves optimization process after finishing
    # 分段损失诊断的频段边界 [Hz]。只写 TB（5_BandLoss/*），不参与 CMA-ES 的 tell()。
    # 设为空 tuple 关闭。段数 = len - 1。
    # 「段号 → 频段」的**内部边界网格**。两端不用管：实际扫频带由 chirp_data.pt
    # 的 "chirp" 字段（老数据则由 des_dof_pos 反解）钉死，落在外面的边界会被丢掉。
    # 以前这两端要手工和 data_collection.py 的 --min/--max_frequency 对齐，已经错位过
    # 一轮（26_09_29 是 0.1-2Hz、26_09_30 是 0.1-4Hz、G1 是 0.1-10Hz，配置恒为 0.1-4）。
    segment_edges_hz: tuple = (0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.8, 4.0, 5.5, 7.0, 10.0)
    # 扫频方式，必须和 data_collection.py 一致："linear" | "log"。
    # 它决定「时间步 ↔ 瞬时频率」的映射，搞错只会让 TB 的频段标签错位，不会报错。
    sweep_kind: str = "linear"

@configclass
class PaceCfg:
    """Overall configuration for Pace Sim2Real task."""
    cmaes: CMAESOptimizerCfg = CMAESOptimizerCfg()

    robot_name: str = MISSING
    data_dir: str = MISSING
    joint_order: list = MISSING
    bounds_params: torch.Tensor = MISSING

##
# Environment configuration
##

@configclass
class PaceSim2realEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PaceSim2realSceneCfg = PaceSim2realSceneCfg()
    # Basic settings
    actions: ActionsCfg = ActionsCfg()  # action = joint position targets (scale = 1.0 -> impedance control)

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim2real: PaceCfg = PaceCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 99999.0  # long episodes
        # viewer settings
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.viewer.eye = (2.0, 2.0, 1.5)
        # simulation settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.sim.render_interval = 4  # render at 100Hz

        self.scene.robot.spawn.articulation_props.fix_root_link = True
