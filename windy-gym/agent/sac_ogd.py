from typing import List, Tuple
import numpy as np
import torch

from agent.sac import TrainerNet as TrainerNetSAC
from agent.core_alg.core_ogd import train_sliding_ogd
from env.base import WindyGym
from torch.utils.tensorboard import SummaryWriter


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, off_policy_random_epochs: int, off_policy_learn_steps: int,
                 tau: float, num_epochs: int, len_buff_eval: int, parallel: bool, ogd_alpha: float, ogd_n: int):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, auto_target_entropy, ent_lr, off_policy_random_epochs,
                                         off_policy_learn_steps, tau, num_epochs, len_buff_eval, parallel)
        self.ogd_past_grads = [[], [], []]
        self.ogd_alpha = ogd_alpha
        self.ogd_n = ogd_n

        self.net_opt_p = None
        self.net_opt_v_1 = None
        self.net_opt_v_2 = None

    def para_lr(self):
        self.monitor.add_scalar('SGD/PolLR', self.ogd_alpha, self.it)
        if self.parallel and self.it >= 5000 and self.it % 1000 == 0:
            self.ogd_alpha = 0.5 * self.ogd_alpha

    def load_file(self, path_or_dict: str or dict):
        if isinstance(path_or_dict, str):
            state_dict = torch.load(path_or_dict, map_location=self.device)
        else:
            state_dict = path_or_dict
        self.policy_net.load_state_dict(state_dict[0])
        self.critic_1_local.load_state_dict(state_dict[1])
        self.critic_2_local.load_state_dict(state_dict[2])
        self.critic_1_target.load_state_dict(state_dict[3])
        self.critic_2_target.load_state_dict(state_dict[4])
        self.ogd_past_grads = state_dict[5]

    def save_file(self, path: str):
        state_dict = [self.policy_net.state_dict(), self.critic_1_local.state_dict(), self.critic_2_local.state_dict(),
                      self.critic_1_target.state_dict(), self.critic_2_target.state_dict(), self.ogd_past_grads]
        torch.save(state_dict, path)

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, np.ndarray,
    np.ndarray]:
        self.para_lr()
        pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2, self.ogd_past_grads = \
            train_sliding_ogd(actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np, self.policy_net,
                              self.critic_1_target, self.critic_2_target, self.critic_1_local, self.critic_2_local,
                              self.net_loss, self.device, self.gamma, self.entropy_factor, self.tau, self.ogd_past_grads,
                              self.ogd_alpha, self.ogd_n)
        self.it += 1
        return pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2
