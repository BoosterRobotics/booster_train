from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "t1_motion_tracking"
