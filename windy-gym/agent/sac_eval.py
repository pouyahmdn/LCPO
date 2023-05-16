import time
from typing import List
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.sac import TrainerNet as TrainerNetSAC
from agent.monitoring.core_log import log_stats_basic
from buffer.buffer import TransitionBuffer
from env.base import WindyGym
from utils.proj_time import ProjectFinishTime


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, off_policy_random_epochs: int, off_policy_learn_steps: int,
                 tau: float, num_epochs: int, len_buff_eval: int, parallel: bool):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, auto_target_entropy, ent_lr, off_policy_random_epochs,
                                         off_policy_learn_steps, tau, num_epochs, len_buff_eval, parallel)
        self.rand_countdown = 0
        self.buff = TransitionBuffer(self.obs_len, self.act_len, batch_size, reward_scale)

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        pi_cpu = self.policy_net.pi(torch.as_tensor(obs, dtype=torch.float, device=self.device)).cpu()
        # array of shape (1, act_dim, act_bins)
        act = (pi_cpu[0, :, :-1].cumsum(-1) <= torch.rand((self.act_len, 1))).sum(dim=-1).numpy()
        return act

    def run_training(self, saved_model: str, num_epochs: int, skip_tb: bool, save_interval: int, eval_interval: int):
        # initialize master from file
        self.load_file(saved_model)

        # initialize parameters
        epoch = 0

        # check elapsed time
        last_time = time.time()
        proj_eta = ProjectFinishTime(num_epochs)

        # initialize env
        # seems like gym objects have no seed
        # self.env.seed(self.seed)
        obs, _ = self.env.reset()

        # training process
        while epoch < num_epochs:
            # collect experience
            batch_ready = False
            self.buff.reset_head()

            while not batch_ready:
                # get policy distribution
                act = self.sample_action(obs)

                next_obs, rew, terminated, truncated, info = self.env.step(act)

                self.buff.add_exp(obs, act, rew, next_obs, terminated, truncated)

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()

                # check if batch is done
                batch_ready = self.buff.buffer_full()

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_stats_basic(self.buff, elapsed, self.monitor, proj_eta, epoch)
            else:
                proj_eta.update_progress(epoch)

            # update counter
            epoch += 1

        self.save_model(epoch)
