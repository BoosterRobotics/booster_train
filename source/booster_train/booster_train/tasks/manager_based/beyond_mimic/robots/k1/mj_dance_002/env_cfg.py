from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster_k1 import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .tracking_env_cfg import TrackingEnvCfg


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.commands.motion.motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_mj2_seg1.npz"
        self.commands.motion.anchor_body_name = "trunk"
        self.commands.motion.body_names = [
            'trunk',
            'aahead_pitch_link',
            'left_hip_roll_link',
            'left_knee_pitch_link',
            'left_ankle_roll_link',
            'right_hip_roll_link',
            'right_knee_pitch_link',
            'right_ankle_roll_link',
            'left_shoulder_roll_link',
            'left_elbow_pitch_link',
            'left_elbow_yaw_link',
            'right_shoulder_roll_link',
            'right_elbow_pitch_link',
            'right_elbow_yaw_link',
        ]


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
        self.scene.terrain.debug_vis = False        # 设为True可视化地形分布
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 每个地形块尺寸（米）
            border_width=20.0,            # 边界宽度（米）
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # 启用课程学习
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None
