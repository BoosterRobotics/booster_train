import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import *
from booster_train.assets.robots.actuator import DelayedImplicitActuatorCfg
from booster_assets import BOOSTER_ASSETS_DIR

ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_4315 = 0.0339552
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_ROB_14 = 0.001

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6416 = 80.
STIFFNESS_4310 = 80.
STIFFNESS_6408 = 80.
STIFFNESS_4315 = 80.
STIFFNESS_ROB_14 = 4.

DAMPING_6416 = 2.0
DAMPING_4310 = 2.0
DAMPING_6408 = 2.0
DAMPING_4315 = 2.0
DAMPING_ROB_14 = 1.

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
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 30.,
                ".*_Hip_Roll": 35.,
                ".*_Hip_Yaw": 20.,
                ".*_Knee_Pitch": 40.,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 8.,
                ".*_Hip_Roll": 12.9,
                ".*_Hip_Yaw": 18.,
                ".*_Knee_Pitch": 12.5,
            },
            stiffness={
                ".*_Hip_Pitch": STIFFNESS_6408,
                ".*_Hip_Roll": STIFFNESS_4315,
                ".*_Hip_Yaw": STIFFNESS_4310,
                ".*_Knee_Pitch": STIFFNESS_6416,
            },
            damping={
                ".*_Hip_Pitch": DAMPING_6408,
                ".*_Hip_Roll": DAMPING_4315,
                ".*_Hip_Yaw": DAMPING_4310,
                ".*_Knee_Pitch": DAMPING_6416,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_6408,
                ".*_Hip_Roll": ARMATURE_4315,
                ".*_Hip_Yaw": ARMATURE_4310,
                ".*_Knee_Pitch": ARMATURE_6416,
            },
        ),
        "feet": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            effort_limit_sim=20.0,
            velocity_limit_sim=18.,
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            stiffness=30,
            damping=2.0,
            armature=2.0 * ARMATURE_4310,
        ),
        "arms": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 14.0,
                ".*_Shoulder_Roll": 14.0,
                ".*_Elbow_Pitch": 14.0,
                ".*_Elbow_Yaw": 14.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            stiffness={
                ".*_Shoulder_Pitch": STIFFNESS_ROB_14,
                ".*_Shoulder_Roll": STIFFNESS_ROB_14,
                ".*_Elbow_Pitch": STIFFNESS_ROB_14,
                ".*_Elbow_Yaw": STIFFNESS_ROB_14,
            },
            damping={
                ".*_Shoulder_Pitch": DAMPING_ROB_14,
                ".*_Shoulder_Roll": DAMPING_ROB_14,
                ".*_Elbow_Pitch": DAMPING_ROB_14,
                ".*_Elbow_Yaw": DAMPING_ROB_14,
            },
            armature={
                ".*_Shoulder_Pitch": ARMATURE_ROB_14,
                ".*_Shoulder_Roll": ARMATURE_ROB_14,
                ".*_Elbow_Pitch": ARMATURE_ROB_14,
                ".*_Elbow_Yaw": ARMATURE_ROB_14,
            },
        ),
        "head": DelayedImplicitActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=6.0,
            velocity_limit_sim=20.0,
            stiffness=4.0,
            damping=1.0,
            armature=0.001,
        ),
    },
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
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 7.33,
                ".*_Shoulder_Roll": 7.33,
                ".*_Elbow_Pitch": 7.33,
                ".*_Elbow_Yaw": 7.33,
            },
            stiffness={
                ".*_Shoulder_Pitch": 50.,
                ".*_Shoulder_Roll": 50.,
                ".*_Elbow_Pitch": 50.,
                ".*_Elbow_Yaw": 50.,
            },
            damping={
                ".*_Shoulder_Pitch": 1.,
                ".*_Shoulder_Roll": 1.,
                ".*_Elbow_Pitch": 1.,
                ".*_Elbow_Yaw": 1.,
            },
            armature={
                ".*_Shoulder_Pitch": ARMATURE_4310,
                ".*_Shoulder_Roll": ARMATURE_4310,
                ".*_Elbow_Pitch": ARMATURE_4310,
                ".*_Elbow_Yaw": ARMATURE_4310,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=25.0,
            velocity_limit_sim=12.57,
            stiffness=200.,
            damping=5.,
            armature=ARMATURE_6408,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.,
                ".*_Hip_Roll": 25.,
                ".*_Hip_Yaw": 25.,
                ".*_Knee_Pitch": 60.,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 16.76,
                ".*_Hip_Roll": 12.57,
                ".*_Hip_Yaw": 12.57,
                ".*_Knee_Pitch": 12.57,
            },
            stiffness={
                ".*_Hip_Pitch": 200.,
                ".*_Hip_Roll": 200.,
                ".*_Hip_Yaw": 200.,
                ".*_Knee_Pitch": 200.,
            },
            damping={
                ".*_Hip_Pitch": 5.,
                ".*_Hip_Roll": 5.,
                ".*_Hip_Yaw": 5.,
                ".*_Knee_Pitch": 5.,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_8112,
                ".*_Hip_Roll": ARMATURE_6408,
                ".*_Hip_Yaw": ARMATURE_6408,
                ".*_Knee_Pitch": ARMATURE_8116,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll"
            ],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50,
            damping=1.0,
            armature=2 * ARMATURE_4315,
        ),
        "head": DelayedImplicitActuatorCfg(
            max_delay=8,
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.0,
            stiffness=10.0,
            damping=1.0,
            armature=0.001,
        ),
    },
)

T1_ACTION_SCALE = {}
for a in BOOSTER_T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

print(f'{BOOSTER_T1_CFG.actuators=}')
print(f'{T1_ACTION_SCALE=}')