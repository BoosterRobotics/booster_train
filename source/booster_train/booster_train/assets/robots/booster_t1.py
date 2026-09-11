import re

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from booster_assets import BOOSTER_ASSETS_DIR

from booster_train.assets.robots import actuator
from booster_train.assets.robots.actuator import (
    BoosterJointDM4310,
    BoosterJointE8112,
    BoosterJointT1E8116,
    BoosterT1AnkleParaWrapperCfg,
)

from booster_train.assets.robots.actuator import (
    BoosterJointE4310,
    BoosterJointE4315,
    BoosterJointE6408,
)

BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.urdf",
        activate_contact_sensors=True,
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": -1.3,
            "right_shoulder_roll_joint": 1.3,
            "left_elbow_yaw_joint": -0.5,
            "right_elbow_yaw_joint": 0.5,
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_pitch_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": actuator.BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            booster_joint_cfgs=BoosterJointE4310(natural_freq=4.0, damping_ratio=1.5),
        ),
        "waist": actuator.BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=["waist_yaw_joint"],
            booster_joint_cfgs=BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
        ),
        "legs": actuator.BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            booster_joint_cfgs={
                ".*_hip_pitch_joint": BoosterJointE8112(natural_freq=4.0, damping_ratio=1.5),
                ".*_hip_roll_joint": BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_hip_yaw_joint": BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_knee_pitch_joint": BoosterJointT1E8116(natural_freq=4.0, damping_ratio=1.5),
            },
        ),
        "feet": actuator.BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            booster_joint_cfgs={
                ".*_ankle_pitch_joint": BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=BoosterJointE4315(),
                    serial_index=0,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                    effort_ratio=(1.7, 0.4),
                    velocity_ratio=(0.6, 1.0),
                    armature_ratio=(2, 0.6),
                    knee_point_velocity_ratio=(1.0, 1.0),
                ),
                ".*_ankle_roll_joint": BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=BoosterJointE4315(),
                    serial_index=1,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                    effort_ratio=(1.7, 0.4),
                    velocity_ratio=(0.6, 1.0),
                    armature_ratio=(2, 0.6),
                    knee_point_velocity_ratio=(1.0, 1.0),
                ),
            },
        ),
        "head": actuator.BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*head.*"],
            booster_joint_cfgs=BoosterJointDM4310(natural_freq=4.0, damping_ratio=1.5),
        ),
    },
)

T1_ACTION_SCALE = {}
per_joint_scale_coefficient = {
    "waist_yaw_joint": 0.16,
    ".*_hip_yaw_joint": 0.28,
    ".*_hip_roll_joint": 0.35,
    ".*_hip_pitch_joint": 0.35,
    ".*_knee_pitch_joint": 0.3,
    ".*_ankle_pitch_joint": 0.17,
    ".*_ankle_roll_joint": 0.28,
    ".*_shoulder_pitch_joint": 0.5,
    ".*_shoulder_roll_joint": 0.5,
    ".*_elbow_pitch_joint": 0.5,
    ".*_elbow_yaw_joint": 0.7,
}

for a in BOOSTER_T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    d = a.damping
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    if not isinstance(d, dict):
        d = {n: d for n in names}

    for n in names:
        if n not in e or n not in s or not s[n]:
            continue

        matched_scale_coefficient = None
        for pattern, scale_coefficient in per_joint_scale_coefficient.items():
            if re.fullmatch(pattern, n):
                matched_scale_coefficient = scale_coefficient
                break

        if matched_scale_coefficient is not None:
            T1_ACTION_SCALE[n] = matched_scale_coefficient * e[n] / s[n]
        else:
            T1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

print(f'{BOOSTER_T1_CFG.actuators=}')
print(f'{T1_ACTION_SCALE=}')
