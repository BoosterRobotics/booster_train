from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions

RPM2RADPS = 2 * torch.pi / 60


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with speed-dependent torque clipping.

    This implements a piecewise-linear torque-speed curve (T-N curve) similar to
    :class:`DelayedImplicitActuator`, but for a PD actuator with delayed commands.
    """

    cfg: BoosterDelayedPDActuatorCfg

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Knee speed for the torque-speed curve. Defaults to velocity_limit (i.e. no reduction).
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        # buffer used for speed-based clipping
        self._joint_vel = torch.zeros_like(self.computed_effort)
        v_knee = self.knee_point_velocity
        v_max = self.velocity_limit
        self._denom = (v_max - v_knee).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Cache measured joint velocity for speed-based torque clipping.
        self._joint_vel[:] = joint_vel
        # Apply delay using the base implementation.
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # Piecewise-linear T-N curve using existing limits:
        # - effort_limit: max torque for |v| <= knee_point_velocity
        # - velocity_limit: torque goes to 0 at |v| == velocity_limit
        joint_vel_abs = self._joint_vel.abs()
        v_max = self.velocity_limit
        tau_max = self.effort_limit

        # If velocity limit is not finite or non-positive, fall back to box limit (or zero).
        non_positive_vmax = v_max <= 0.0
        non_finite_vmax = ~torch.isfinite(v_max)

        # Avoid division-by-zero when v_max == v_knee for some joints.
        tau_linear = tau_max * (v_max - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)

        max_effort = torch.where(non_finite_vmax, tau_max, max_effort)
        max_effort = torch.where(non_positive_vmax, torch.zeros_like(max_effort), max_effort)

        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterJointCfg:
    """Configuration for booster joint models."""

    joint_model_name: str = MISSING

    effort_limit: float = MISSING
    velocity_limit: float = MISSING
    knee_point_velocity: float = MISSING
    armature: float = MISSING

    stiffness: float = None
    damping: float = None

    natural_freq: float = 10.     # 10Hz
    '''
    Natural frequency (Hz) for computing default stiffness and damping.
    '''
    damping_ratio: float = 2.0
    '''
    Damping ratio for computing default damping.
    '''

    def __post_init__(self):
        if self.stiffness is None:
            self.stiffness = self.armature * (2 * 3.1415926535 * self.natural_freq)**2
        if self.damping is None:
            self.damping = 2 * self.damping_ratio * self.armature * (2 * 3.1415926535 * self.natural_freq)


@configclass
class BoosterDelayedActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`BoosterDelayedPDActuator`."""

    class_type: type = MISSING

    knee_point_velocity: dict[str, float] | float | None = None
    """Knee speed for the torque-speed curve.

    If None, defaults to :attr:`velocity_limit` (no speed-dependent reduction).
    """

    stiffness: dict[str, float] | float | None = None
    damping: dict[str, float] | float | None = None

    booster_joint_cfgs: dict[str, BoosterJointCfg] | BoosterJointCfg | None = None
    effort_limit_sim_multiplier: float = 1.0
    velocity_limit_sim_multiplier: float = 1.0

    def __post_init__(self):
        if self.booster_joint_cfgs is not None:
            if isinstance(self.booster_joint_cfgs, BoosterJointCfg):
                self.effort_limit = self.booster_joint_cfgs.effort_limit
                self.effort_limit_sim = self.booster_joint_cfgs.effort_limit * self.effort_limit_sim_multiplier
                self.velocity_limit = self.booster_joint_cfgs.velocity_limit
                self.velocity_limit_sim = self.booster_joint_cfgs.velocity_limit * self.velocity_limit_sim_multiplier
                self.knee_point_velocity = self.booster_joint_cfgs.knee_point_velocity
                self.armature = self.booster_joint_cfgs.armature
                if self.stiffness is None:
                    self.stiffness = self.booster_joint_cfgs.stiffness
                if self.damping is None:
                    self.damping = self.booster_joint_cfgs.damping

            elif isinstance(self.booster_joint_cfgs, dict):
                self.effort_limit = {
                    joint_name: joint_cfg.effort_limit for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                }
                self.effort_limit_sim = {
                    joint_name: joint_cfg.effort_limit * self.effort_limit_sim_multiplier
                    for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                }
                self.velocity_limit = {
                    joint_name: joint_cfg.velocity_limit for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                }
                self.velocity_limit_sim = {
                    joint_name: joint_cfg.velocity_limit * self.velocity_limit_sim_multiplier
                    for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                }
                self.armature = {
                    joint_name: joint_cfg.armature for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                }
                self.knee_point_velocity = {
                    joint_name: joint_cfg.knee_point_velocity for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                }
                if self.stiffness is None:
                    self.stiffness = {
                        joint_name: joint_cfg.stiffness for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                    }
                if self.damping is None:
                    self.damping = {
                        joint_name: joint_cfg.damping for joint_name, joint_cfg in self.booster_joint_cfgs.items()
                    }


@configclass
class BoosterDelayedPDActuatorCfg(BoosterDelayedActuatorCfg):
    """Configuration for :class:`BoosterDelayedPDActuator`."""

    class_type: type = BoosterDelayedPDActuator


@configclass
class BoosterDelayedImplicitActuatorCfg(BoosterDelayedActuatorCfg):
    """Configuration for :class:`BoosterDelayedPDActuator`."""

    class_type: type = DelayedImplicitActuator


@configclass
class ParallelJointWrapperCfg(BoosterJointCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration.
    """

    joint_model_name: str = "ParallelJointWrapper"

    effort_ratio: tuple[float, float] = MISSING
    velocity_ratio: tuple[float, float] = MISSING
    armature_ratio: tuple[float, float] = MISSING
    knee_point_velocity_ratio: tuple[float, float] = (1.0, 1.0)

    base_joint_cfg: BoosterJointCfg = MISSING
    serial_index: int = MISSING     # 0 for pitch, 1 for roll

    def __post_init__(self):
        self.effort_limit = self.effort_ratio[self.serial_index] * self.base_joint_cfg.effort_limit
        self.velocity_limit = self.velocity_ratio[self.serial_index] * self.base_joint_cfg.velocity_limit
        self.knee_point_velocity = self.knee_point_velocity_ratio[self.serial_index] * self.base_joint_cfg.knee_point_velocity
        self.armature = self.armature_ratio[self.serial_index] * self.base_joint_cfg.armature
        self.joint_model_name = f"{self.joint_model_name}({self.base_joint_cfg.joint_model_name})[{self.serial_index}]"
        super().__post_init__()


@configclass
class BoosterK1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration for K1 ankle.
    """

    joint_model_name: str = "BoosterK1AnkleParaWrapper"

    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)


@configclass
class BoosterT1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration for T1 ankle.
    """

    joint_model_name: str = "BoosterT1AnkleParaWrapper"

    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)


@configclass
class BoosterJointE8116(BoosterJointCfg):
    """Configuration for E8116 booster joint model."""

    joint_model_name: str = "E8116"

    effort_limit: float = 130
    velocity_limit: float = 14.66
    knee_point_velocity: float = 6.28
    armature: float = 0.0636012


# K1 Hip Pitch / T1 Waist, Hip Roll & Yaw
@configclass
class BoosterJointE6408(BoosterJointCfg):
    """Configuration for E6408 booster joint model."""

    joint_model_name: str = "E6408"

    effort_limit: float = 68.0
    velocity_limit: float = 14.66
    knee_point_velocity: float = 1.88
    armature: float = 0.0478125


# K1 Hip Roll
@configclass
class BoosterJointE4315(BoosterJointCfg):
    """Configuration for E4315 booster joint model."""

    joint_model_name: str = "E4315"

    effort_limit: float = 43.0
    velocity_limit: float = 12.15
    knee_point_velocity: float = 8.38
    armature: float = 0.0339552


# K1 Hip Yaw & T1 Arm
@configclass
class BoosterJointE4310(BoosterJointCfg):
    """Configuration for E4310 booster joint model."""

    joint_model_name: str = "E4310"

    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0282528


# K1 Knee
@configclass
class BoosterJointE6416(BoosterJointCfg):
    """Configuration for E6416 booster joint model."""

    joint_model_name: str = "E6416"

    effort_limit: float = 112.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.09
    armature: float = 0.095625


# K1 Arm
@configclass
class BoosterJointR14(BoosterJointCfg):
    """Configuration for R14 booster joint model."""

    joint_model_name: str = "R14"

    effort_limit: float = 14
    velocity_limit: float = 33.51
    knee_point_velocity: float = 5.24
    armature: float = 0.001


# K1 Neck
@configclass
class BoosterJointHT4438(BoosterJointCfg):
    """Configuration for HT4438 booster joint model."""

    joint_model_name: str = "HT4438"

    effort_limit: float = 6.0
    velocity_limit: float = 7.85
    knee_point_velocity: float = 10.47
    armature: float = 0.001


# T1 joint models
@configclass
class BoosterJointE8112(BoosterJointCfg):
    """Configuration for the T1 E8112 booster joint model."""

    joint_model_name: str = "E8112"
    effort_limit: float = 98.8
    velocity_limit: float = 15.29
    knee_point_velocity: float = 7.53
    armature: float = 0.0523908


@configclass
class BoosterJointT1E8116(BoosterJointCfg):
    """Configuration for the T1 E8116 booster joint model."""

    joint_model_name: str = "E8116"
    effort_limit: float = 130.5
    velocity_limit: float = 14.76
    knee_point_velocity: float = 6.71
    armature: float = 0.0636012


@configclass
class BoosterJointDM4310(BoosterJointCfg):
    """Configuration for the T1 DM4310 booster joint model."""

    joint_model_name: str = "DM4310"
    effort_limit: float = 7.0
    velocity_limit: float = 41.89
    knee_point_velocity: float = 26.18
    armature: float = 0.0018


# T2 joint models. T2 actuator data is specified in RPM and converted to rad/s.
@configclass
class BoosterJointB4522_35(BoosterJointCfg):
    """Configuration for the T2 B4522-35 booster joint model."""

    joint_model_name: str = "B4522-35"
    effort_limit: float = 22
    velocity_limit: float = 177 * RPM2RADPS
    knee_point_velocity: float = 122 * RPM2RADPS
    armature: float = 0.007209
    natural_freq: float = 8


@configclass
class BoosterJointB6030_24(BoosterJointCfg):
    """Configuration for the T2 B6030-24 booster joint model."""

    joint_model_name: str = "B6030-24"
    effort_limit: float = 74
    velocity_limit: float = 224 * RPM2RADPS
    knee_point_velocity: float = 110 * RPM2RADPS
    armature: float = 0.017
    natural_freq: float = 8


@configclass
class BoosterJointB6025_24(BoosterJointCfg):
    """Configuration for the T2 B6025-24 booster joint model."""

    joint_model_name: str = "B6025-24"
    effort_limit: float = 60
    velocity_limit: float = 227 * RPM2RADPS
    knee_point_velocity: float = 123 * RPM2RADPS
    armature: float = 0.0137
    natural_freq: float = 8


@configclass
class BoosterJointB7629_22(BoosterJointCfg):
    """Configuration for the T2 B7629-22 booster joint model."""

    joint_model_name: str = "B7629-22"
    effort_limit: float = 135
    velocity_limit: float = 200 * RPM2RADPS
    knee_point_velocity: float = 59 * RPM2RADPS
    armature: float = 0.0281
    natural_freq: float = 8


@configclass
class BoosterT2WaistParaWrapperCfg(ParallelJointWrapperCfg):
    """Parallel joint wrapper configuration for the T2 waist."""

    joint_model_name: str = "BoosterT2WaistParaWrapper"
    effort_ratio: tuple[float, float] = (2.3, 2.4)
    velocity_ratio: tuple[float, float] = (0.5, 0.4)
    armature_ratio: tuple[float, float] = (2.884254, 3.825669)
    knee_point_velocity_ratio: tuple[float, float] = (0.5, 0.4)
    natural_freq: float = 8


@configclass
class BoosterT2AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Parallel joint wrapper configuration for the T2 ankle."""

    joint_model_name: str = "BoosterT2AnkleParaWrapper"
    effort_ratio: tuple[float, float] = (2.2, 1.2)
    velocity_ratio: tuple[float, float] = (0.7, 1.0)
    armature_ratio: tuple[float, float] = (2.620008, 0.9734019)
    knee_point_velocity_ratio: tuple[float, float] = (0.7, 1.0)
    natural_freq: float = 8
