from typing import Tuple, List
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as TrainerNetA2C
from agent.core_alg.core_lcppo import train_lcppo
from buffer.buffer_ood import OutOfDSampler
from env.base import WindyGym


class TrainerNet(TrainerNetA2C):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float, lam: float,
                 auto_target_entropy: float, ent_lr: float, len_buff_eval: int, ppo_kl: float, ppo_iters: int,
                 ppo_clip: float, kappa: float, ood_mini_len: int, ood_len: int, parallel: bool):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, lam, auto_target_entropy, ent_lr, len_buff_eval, parallel)
        self.ppo_kl = ppo_kl
        self.ppo_iters = ppo_iters
        self.ppo_clip = ppo_clip
        self.lcppo_kappa = kappa
        self.batch_rng = np.random.RandomState(seed=seed)
        self.ood_buf = OutOfDSampler(self.obs_len, ood_mini_len, ood_len, environment.is_different)
        self.ood_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                  np.ndarray, float]:
        self.para_lr()
        self.ood_buf.add_many_exp(obs_np)
        ood_obs = self.ood_buf.get(self.batch_rng, len(obs_np))
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_lcppo(self.value_net, self.policy_net, self.net_opt_p, self.net_opt_v, self.net_loss, self.ood_loss,
                        self.device, actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np, self.gamma,
                        self.lamb, self.entropy_factor, self.ppo_kl, self.lcppo_kappa, self.ppo_iters, self.ppo_clip,
                        ood_obs_np=np.array(ood_obs), monitor=self.monitor, it=self.it)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min
