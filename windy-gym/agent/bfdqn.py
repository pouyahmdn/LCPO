from typing import List, Tuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.dqn import TrainerNet as TrainerNetDQN
from agent.core_alg.core_bfdqn import train_bfdqn
from buffer.buffer import TransitionBuffer

from env.base import WindyGym


class TrainerNet(TrainerNetDQN):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, eps_decay: float,
                 eps_min: float, num_epochs: int, lr_rate: float, gamma: float, off_policy_random_epochs: int,
                 off_policy_learn_steps: int, tau: float, bf_n: int, bf_g: float, bf_buff_len: int, len_buff_eval: int,
                 parallel: bool):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, eps_decay, eps_min, num_epochs, lr_rate, gamma,
                                         off_policy_random_epochs, off_policy_learn_steps, tau, len_buff_eval, parallel)

        self.buff = TransitionBuffer(self.obs_len, self.act_len, bf_buff_len, reward_scale,
                                     size_fifo=off_policy_learn_steps)

        self.bf_n = bf_n
        self.bf_vars = None
        self.bf_g = bf_g / np.power(2, np.arange(self.bf_n))
        self.bf_c = np.power(2, np.arange(self.bf_n))

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        self.para_lr()
        v_loss, q_val, trg_val, self.bf_vars = train_bfdqn(actions_np, next_obs_np, rewards_np, obs_np, term_np,
                                                           trunc_np, self.q_net, self.target_net, self.net_opt_q,
                                                           self.net_loss, self.device, self.gamma, self.tau, self.bf_n,
                                                           self.bf_g, self.bf_c, self.bf_vars, self.it)
        self.it += 1
        return v_loss, q_val, trg_val
