import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from booster_assets import BOOSTER_ASSETS_DIR

from booster_train.assets.robots import actuator
from booster_train.assets.robots.actuator import BoosterDelayedPDActuatorCfg


BOOSTER_T2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/T2/T2_31dof/T2_31dof.urdf",
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
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": -1.3,
            "right_shoulder_roll_joint": 1.3,
            "left_elbow_yaw_joint": -0.5,
            "right_elbow_yaw_joint": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=0,
            effort_limit_sim_multiplier=1.2,
            velocity_limit_sim_multiplier=5.0,
            joint_names_expr=["aa_head_yaw_joint", "head_pitch_joint"],
            booster_joint_cfgs=actuator.BoosterJointB4522_35(),
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=0,
            effort_limit_sim_multiplier=1.2,
            velocity_limit_sim_multiplier=5.0,
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
            ],
            booster_joint_cfgs={
                ".*_shoulder_pitch_joint": actuator.BoosterJointB6030_24(),
                ".*_shoulder_roll_joint": actuator.BoosterJointB6025_24(),
                ".*_elbow_pitch_joint": actuator.BoosterJointB6025_24(),
                ".*_elbow_yaw_joint": actuator.BoosterJointB6025_24(),
                ".*_wrist_pitch_joint": actuator.BoosterJointB4522_35(),
                ".*_wrist_yaw_joint": actuator.BoosterJointB4522_35(),
                ".*_wrist_roll_joint": actuator.BoosterJointB4522_35(),
            },
        ),
        "waist": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=0,
            effort_limit_sim_multiplier=1.2,
            velocity_limit_sim_multiplier=5.0,
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"],
            booster_joint_cfgs={
                "waist_pitch_joint": actuator.BoosterT2WaistParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointB6025_24(),
                    serial_index=0,
                ),
                "waist_roll_joint": actuator.BoosterT2WaistParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointB6025_24(),
                    serial_index=1,
                ),
                "waist_yaw_joint": actuator.BoosterJointB7629_22(),
            },
        ),
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=0,
            effort_limit_sim_multiplier=1.2,
            velocity_limit_sim_multiplier=5.0,
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            booster_joint_cfgs={
                ".*_hip_pitch_joint": actuator.BoosterJointB7629_22(),
                ".*_hip_roll_joint": actuator.BoosterJointB7629_22(),
                ".*_hip_yaw_joint": actuator.BoosterJointB7629_22(),
                ".*_knee_pitch_joint": actuator.BoosterJointB7629_22(),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=0,
            effort_limit_sim_multiplier=1.2,
            velocity_limit_sim_multiplier=5.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            booster_joint_cfgs={
                ".*_ankle_pitch_joint": actuator.BoosterT2AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointB6025_24(),
                    serial_index=0,
                ),
                ".*_ankle_roll_joint": actuator.BoosterT2AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointB6025_24(),
                    serial_index=1,
                ),
            },
        ),
    },
)

T2_ACTION_SCALE = {}
for actuator_cfg in BOOSTER_T2_CFG.actuators.values():
    effort = actuator_cfg.effort_limit
    stiffness = actuator_cfg.stiffness
    joint_names = actuator_cfg.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for joint_name in joint_names:
        if joint_name in effort and joint_name in stiffness and stiffness[joint_name]:
            T2_ACTION_SCALE[joint_name] = 0.25 * effort[joint_name] / stiffness[joint_name]
