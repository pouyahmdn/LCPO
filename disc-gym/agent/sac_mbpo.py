import time
from typing import List
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_sac
from agent.sac import TrainerNet as TrainerNetSAC
from env.base import DiscGym
from agent.core_alg.core_mbpo import MBPO
from utils.proj_time import ProjectFinishTime


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: DiscGym, environment_eval: DiscGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, off_policy_random_epochs: int, off_policy_learn_steps: int,
                 tau: float, num_epochs: int, len_buff_eval: int, mbpo_warm_up: int):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, auto_target_entropy, ent_lr, off_policy_random_epochs,
                                         off_policy_learn_steps, tau, num_epochs, len_buff_eval)

        self.deep_mbpo = MBPO(state_dim=self.obs_len,
                              action_dim=self.act_len * self.act_bins,
                              memory_capacity=num_epochs * off_policy_learn_steps)
        self.mbpo_warm_up = mbpo_warm_up

    def sample_many_actions(self, obs: np.ndarray) -> np.ndarray:
        pi_cpu = self.policy_net.pi(torch.as_tensor(obs, dtype=torch.float, device=self.device)).cpu()
        # array of shape (BATCH, act_dim, act_bins)
        acts = (pi_cpu[:, :, :-1].cumsum(-1) <= torch.rand((obs.shape[0], self.act_len, 1))).sum(dim=-1).numpy()
        return acts

    def run_training(self, saved_model: str, num_epochs: int, skip_tb: bool, save_interval: int, eval_interval: int):
        # initialize master from file
        if saved_model is not None:
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
            self.buff.reset_head()

            for _ in range(self.off_policy_learn_steps):
                act = self.sample_action(obs)

                next_obs, rew, terminated, truncated, info = self.env.step(act)

                self.buff.add_exp(obs, act, rew, next_obs, terminated, truncated)

                act_1hot = np.zeros((self.act_len, self.act_bins))
                for i in range(self.act_len):
                    act_1hot[i, act[i]] = 1
                act_1hot = act_1hot.ravel()

                self.deep_mbpo.add_experience(obs.copy(), act_1hot, rew, next_obs.copy(), terminated)

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()

            self.deep_mbpo.train()

            if epoch >= self.mbpo_warm_up:
                all_states, _, _, _, all_term, all_trunc = self.buff.get_batch(self.batch_rng, self.batch_size)

                all_fake_acts = self.sample_many_actions(all_states)
                all_fake_next_states, all_fake_rewards = self.deep_mbpo.predict(all_states, all_fake_acts)

                all_fake_n_rewards = all_fake_rewards / np.sqrt(self.ret_rms.var + 1)

                # Train SAC
                pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2 = \
                    self.train(all_fake_acts, all_fake_next_states, all_fake_n_rewards, all_states, all_term, all_trunc)

                self.ret_rms.update(q_target)

                norm_entropy = (real_entropy - self.minimum_entropy) / (self.maximum_entropy - self.minimum_entropy)
                self.tune_entropy(all_states)

            else:
                v_loss_1, v_loss_2, pg_loss, norm_entropy = np.nan, np.nan, np.nan, np.nan
                q_target, q_local_1, q_local_2 = np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_sac(self.buff, q_target, q_local_1,
                        q_local_2, pg_loss, v_loss_1, v_loss_2, self.entropy_factor, norm_entropy, elapsed,
                        self.monitor, proj_eta, epoch)
            else:
                proj_eta.update_progress(epoch)

            # save model
            if epoch % save_interval == 0:
                self.save_model(epoch)

            # evaluate model
            if epoch % eval_interval == 0:
                self.evaluate(epoch, skip_tb)

            # update counter
            epoch += 1

        self.save_model(epoch)
