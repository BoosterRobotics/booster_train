# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from pathlib import Path

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from booster_train.assets.robots.booster_t2 import BOOSTER_T2_CFG as ROBOT_CFG, T2_ACTION_SCALE

from .tracking_env_cfg import TrackingEnvCfg


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T2_ACTION_SCALE
        self.scene.terrain.use_terrain_origins = True
        self.commands.motion.motion_file = "t2_dance.npz"
        self.commands.motion.anchor_body_name = "trunk"
        self.commands.motion.body_names = [
            "trunk",
            "waist_yaw_link",
            "head_pitch_link",
            "left_hip_roll_link",
            "left_knee_pitch_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_pitch_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_roll_link",
        ]
        for group in (self.observations.policy, self.observations.critic):
            for term in group.__dict__.values():
                if hasattr(term, "clip"):
                    term.clip = (-100.0, 100.0)


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False  # Set to True to visualize the terrain distribution.
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),  # Size of each terrain tile in meters.
            border_width=20.0,  # Border width in meters.
            num_rows=5,  # Number of terrain grid rows.
            num_cols=10,  # Number of terrain grid columns.
            horizontal_scale=0.1,  # Horizontal resolution.
            vertical_scale=0.005,  # Vertical resolution.
            slope_threshold=0.75,  # Mesh simplification threshold.
            use_cache=False,  # Regenerate the terrain each time.
            curriculum=False,  # Disable curriculum learning.
            sub_terrains={
                # 80% nearly flat, very smooth terrain.
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),  # Height variation of 0-0.5 cm, nearly flat.
                    noise_step=0.005,  # Noise step of 0.5 cm.
                    border_width=0.25,
                ),
                # 20% randomly rough terrain.
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),  # Height variation of +/-1.5 cm.
                    noise_step=0.005,  # Noise step of 0.5 cm.
                    border_width=0.25,
                ),
            },
        )


@configclass
class PlayRoughWoStateEstimationEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.commands.motion.play = True
        self.events.push_robot = None


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.commands.motion.play = True
        self.events.push_robot = None
