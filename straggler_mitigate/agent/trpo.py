from typing import Tuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as TrainerNetA2C
from agent.core_alg.core_trpo import train_trpo
from cenv.clb.pyenv import PyLoadBalanceEnv
from param import config


class TrainerNet(TrainerNetA2C):
    def __init__(self, environment: PyLoadBalanceEnv, monitor: SummaryWriter, output_folder: str):
        super(TrainerNet, self).__init__(environment, monitor, output_folder)
        self.batch_rng = np.random.RandomState(seed=config.seed)
        self.it = 0

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              times_np: np.ndarray, dones_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                   np.ndarray, float]:
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_trpo(self.value_net, self.policy_net, self.net_opt_v, self.net_loss, self.device,
                       actions_np, next_obs_np, rewards_np, obs_np, dones_np, obs_np[:, [-1]], self.gamma_rate,
                       self.entropy_factor, monitor=self.monitor, it=self.it, times_np=times_np)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min
