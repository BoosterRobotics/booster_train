import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import *
from booster_train.assets.robots import actuator
from booster_train.assets.robots.actuator import (
    BoosterDelayedImplicitActuatorCfg,
    BoosterDelayedPDActuatorCfg,
    DelayedImplicitActuatorCfg
)

from booster_assets import BOOSTER_ASSETS_DIR

BOOSTER_K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.urdf",
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
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            "left_shoulder_roll_joint": -1.3,
            "right_shoulder_roll_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            booster_joint_cfgs={
                ".*_hip_pitch_joint": actuator.BoosterJointE6408(natural_freq = 4.0, damping_ratio = 1.5),
                ".*_hip_roll_joint": actuator.BoosterJointE4315(natural_freq = 4.0, damping_ratio = 1.5),
                ".*_hip_yaw_joint": actuator.BoosterJointE4310(natural_freq = 4.0, damping_ratio = 1.5),
                ".*_knee_pitch_joint": actuator.BoosterJointE6416(natural_freq = 4.0, damping_ratio = 1.0),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            booster_joint_cfgs={
                ".*_ankle_pitch_joint": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=0,
                    natural_freq = 4.0,
                    damping_ratio = 1.5,
                    armature_ratio=(1.4, 0.4),
                ),
                ".*_ankle_roll_joint": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=1,
                    natural_freq = 4.0,
                    damping_ratio = 1.5,
                    armature_ratio=(1.4, 0.4),
                ),
            },
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            booster_joint_cfgs=actuator.BoosterJointR14(),
        ),
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*head.*"],
            booster_joint_cfgs=actuator.BoosterJointHT4438(),
        ),
    }
)

K1_ACTION_SCALE = {}
for a in BOOSTER_K1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            K1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

print(f'{BOOSTER_K1_CFG.actuators=}')
print(f'{K1_ACTION_SCALE=}')
