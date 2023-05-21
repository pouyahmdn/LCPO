from typing import Dict, Tuple, Callable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from agent.a2c import TrainerNet as TrainerNetA2C
from agent.core_alg.core_lcpo import train_trpo
from buffer.buffer_ood import OutOfDSampler
from cenv.clb.pyenv import PyLoadBalanceEnv
from neural_net.nn_perm import PermInvNet
from param import config


def make_func(threshold: float) -> Callable:
    def state_is_different(data: np.ndarray, base: np.ndarray) -> np.ndarray:
        mu, cov = np.mean(base[:, 27:29], axis=0), np.cov(base[:, 27:29], rowvar=False)
        cen_dat = data[:, 27:29] - mu
        ll = -(cen_dat @ np.linalg.inv(cov) * cen_dat).mean(axis=-1) / 2
        return ll < threshold
    return state_is_different


class TrainerNet(TrainerNetA2C):
    def __init__(self, environment: PyLoadBalanceEnv, monitor: SummaryWriter, output_folder: str):
        super(TrainerNet, self).__init__(environment, monitor, output_folder)
        self.old_policy_net = torch.jit.script(PermInvNet(self.obs_len, self.act_len,
                                               config.num_servers, self.aux_state).to(self.device))
        self.ood_buf = OutOfDSampler(self.obs_len, config.master_batch*5, config.master_batch*config.num_epochs,
                                     make_func(config.lcpo_thresh))
        self.batch_rng = np.random.RandomState(seed=config.seed)

        self.local_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.it = 0

    def load_file(self, path_or_dict: str or Dict):
        if isinstance(path_or_dict, str):
            state_dict = torch.load(path_or_dict, map_location=self.device)
        else:
            state_dict = path_or_dict
        self.policy_net.load_state_dict(state_dict[0])
        self.value_net.load_state_dict(state_dict[1])
        self.net_opt_p.load_state_dict(state_dict[2])
        self.net_opt_v.load_state_dict(state_dict[3])

    def save_file(self, path: str):
        state_dict = [self.policy_net.state_dict(), self.value_net.state_dict(),
                      self.net_opt_p.state_dict(), self.net_opt_v.state_dict()]
        torch.save(state_dict, path)

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              times_np: np.ndarray, dones_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                   np.ndarray, float]:
        new_obs_s = obs_np[obs_np[:, -1] < 0.5]
        self.ood_buf.add_many_exp(new_obs_s)
        ood_obs = self.ood_buf.get(self.batch_rng, len(obs_np))
        nt = torch.get_num_threads()
        torch.set_num_threads(3)
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_trpo(self.value_net, self.policy_net, self.net_opt_p, self.net_opt_v, self.net_loss, self.device,
                       actions_np, next_obs_np, rewards_np, obs_np, dones_np, obs_np[:, [-1]], self.gamma_rate,
                       self.entropy_factor, ood_obs_np=np.array(ood_obs), monitor=self.monitor, it=self.it,
                       times_np=times_np)
        torch.set_num_threads(nt)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min
