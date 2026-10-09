# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotor
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import DelayBuffer
if TYPE_CHECKING:
    # only for type checking
    from .pace_actuator_cfg import PaceDCMotorCfg

class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with encoder bias and action delay.

    The actuator models a DC motor whose controller receives joint positions in the encoder
    frame using q_encoder = q_physical - encoder_bias. In other words,
    the controller operates on biased (encoder) positions rather than the true joint positions.

    The torque command computed by the PD controller is applied after a configurable delay
    (in simulation steps) to represent latency between command calculation and actuation.

    The software implementation is inspired by DelayedPDActuator.
    """

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)

        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.torques_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Controller feedback uses q_encoder = q_physical - encoder_bias.
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        self.applied_effort = control_action_sim.joint_efforts
        return control_action_sim


class PaceDCMotorCmdDelay(PaceDCMotor):
    """Delay reference commands before PD; feedback and velocity limits use current state."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.command_buffers = {
            name: DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
            for name in ("joint_positions", "joint_velocities", "joint_efforts")
        }
        self.update_time_lags(cfg.max_delay)

    def update_time_lags(self, delay, env_ids=None):
        if isinstance(delay, torch.Tensor):
            delay = delay.flatten()
        for buffer in self.command_buffers.values():
            buffer.set_time_lag(delay, env_ids)

    def reset(self, env_ids):
        super().reset(env_ids)
        for buffer in self.command_buffers.values():
            buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        for name, buffer in self.command_buffers.items():
            value = getattr(control_action, name)
            if value is not None:
                setattr(control_action, name, buffer.compute(value))
        # Bypass PaceDCMotor.compute: do not also delay the PD output torque.
        return DCMotor.compute(self, control_action, joint_pos - self.encoder_bias, joint_vel)
