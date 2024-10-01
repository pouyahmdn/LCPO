from typing import List, Tuple
import numpy as np

from agent.sac import TrainerNet as TrainerNetSAC
from agent.core_alg.core_ewc import train_sac_ewc
from env.base import WindyGym
from torch.utils.tensorboard import SummaryWriter


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, off_policy_random_epochs: int, off_policy_learn_steps: int,
                 tau: float, num_epochs: int, len_buff_eval: int, parallel: bool, ewc_alpha: float, ewc_gamma: float):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, auto_target_entropy, ent_lr, off_policy_random_epochs,
                                         off_policy_learn_steps, tau, num_epochs, len_buff_eval, parallel)
        self.ewc_importances = [{}, {}, {}]
        self.ewc_past_weights = [{}, {}, {}]
        self.ewc_alpha = ewc_alpha
        self.ewc_gamma = ewc_gamma

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, np.ndarray,
    np.ndarray]:
        self.para_lr()
        pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2, self.ewc_importances, self.ewc_past_weights = \
            train_sac_ewc(actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np, self.policy_net,
                          self.critic_1_target, self.critic_2_target, self.critic_1_local, self.critic_2_local,
                          self.net_opt_v_1, self.net_opt_v_2, self.net_opt_p, self.net_loss, self.device, self.gamma,
                          self.entropy_factor, self.tau, self.ewc_importances, self.ewc_past_weights, self.ewc_alpha,
                          self.ewc_gamma)
        self.it += 1
        return pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2
