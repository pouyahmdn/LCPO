from typing import Tuple, List
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from agent.a2c import TrainerNet as TrainerNetA2C
from agent.core_alg.core_ppo import train_ppo
from env.base import DiscGym


class TrainerNet(TrainerNetA2C):
    def __init__(self, environment: DiscGym, environment_eval: DiscGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float, lam: float,
                 auto_target_entropy: float, ent_lr: float, len_buff_eval: int, ppo_kl: float, ppo_iters: int,
                 ppo_clip: float):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, lam, auto_target_entropy, ent_lr, len_buff_eval)
        self.ppo_kl = ppo_kl
        self.ppo_iters = ppo_iters
        self.ppo_clip = ppo_clip

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                  np.ndarray, float]:
        self.para_lr()
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_ppo(self.value_net, self.policy_net, self.net_opt_p, self.net_opt_v, self.net_loss, self.device,
                      actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np, self.gamma, self.lamb,
                      self.entropy_factor, self.ppo_kl, self.ppo_iters, self.ppo_clip, monitor=self.monitor, it=self.it)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min
