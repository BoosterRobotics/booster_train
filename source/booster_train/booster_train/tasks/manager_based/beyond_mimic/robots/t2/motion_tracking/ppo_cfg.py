from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "t2_motion_tracking"

    def __post_init__(self):
        super().__post_init__()

        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
