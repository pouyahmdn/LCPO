from typing import Tuple

import torch
import numpy as np

from agent.core_alg.core_ewc import train_sac_ewc
from agent.sac import TrainerNet as TrainerNetSAC
from cenv.clb.pyenv import PyLoadBalanceEnv
from param import config
from torch.utils.tensorboard import SummaryWriter


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: PyLoadBalanceEnv, monitor: SummaryWriter, output_folder: str):
        super(TrainerNet, self).__init__(environment, monitor, output_folder)
        self.ewc_importances = [{}, {}, {}]
        self.ewc_past_weights = [{}, {}, {}]
        self.ewc_alpha = config.ewc_alpha
        self.ewc_gamma = config.ewc_gamma

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              times_np: np.ndarray, dones_np: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2, self.ewc_importances, self.ewc_past_weights = \
            train_sac_ewc(actions_np, next_obs_np, rewards_np, obs_np, dones_np, self.policy_net,
                          self.critic_1_target, self.critic_2_target, self.critic_1_local,
                          self.critic_2_local, self.net_opt_v_1, self.net_opt_v_2, self.net_opt_p,
                          self.net_loss, self.device, self.gamma_rate, self.entropy_factor, obs_np[:, -1],
                          next_obs_np[:, -1], self.max_act, self.ewc_importances, self.ewc_past_weights, self.ewc_alpha,
                          self.ewc_gamma, times_np)
        return pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2
