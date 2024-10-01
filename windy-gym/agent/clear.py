import time
from typing import List, Tuple
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as TrainerNetA2C
from agent.monitoring.core_log import log_a2c, print_eval, log_stats_core
from agent.core_alg.core_clear import train_clear
from buffer.buffer_clear import TransitionBufferClear
from utils.proj_time import ProjectFinishTime

from env.base import WindyGym


class TrainerNet(TrainerNetA2C):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, clear_c: float, clear_rho: float, policy_clone_coeff: float,
                 value_clone_coeff: float, num_epochs: int, len_buff_eval: int, parallel: bool):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, 1, auto_target_entropy, ent_lr, len_buff_eval, parallel)

        self.batch_size = batch_size
        self.buff = TransitionBufferClear(self.obs_len, self.act_len, (num_epochs + 1) * batch_size, reward_scale,
                                          size_fifo=batch_size)
        self.batch_rng = np.random.RandomState(seed=self.seed)
        self.buff_eval = TransitionBufferClear(self.obs_len, self.act_len, len_buff_eval, reward_scale,
                                               size_fifo=batch_size)

        assert policy_clone_coeff > 0
        assert value_clone_coeff > 0
        self.clear_c = clear_c
        self.clear_rho = clear_rho
        self.policy_clone_coeff = policy_clone_coeff
        self.value_clone_coeff = value_clone_coeff

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            a, log_prob = self.policy_net.sample_action_prob(torch.as_tensor(obs, dtype=torch.float, device=self.device))
            value_s = self.value_net.forward(torch.as_tensor(obs, dtype=torch.float, device=self.device)).cpu().squeeze()
        return a.numpy(), log_prob.numpy(), value_s.numpy()

    def train(self, actions_np: np.ndarray, action_log_probs: np.ndarray, action_values: np.ndarray,
              next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray, term_np: np.ndarray,
              trunc_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float]:
        self.para_lr()
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_clear(self.value_net, self.policy_net, self.net_opt_p, self.net_opt_v, self.net_loss, self.device,
                        actions_np, action_log_probs, action_values, next_obs_np, rewards_np, obs_np, term_np, trunc_np,
                        self.gamma, self.entropy_factor, self.clear_c, self.clear_rho, self.policy_clone_coeff,
                        self.value_clone_coeff, monitor=self.monitor, it=self.it)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min

    def run_training(self, saved_model: str, num_epochs: int, skip_tb: bool, save_interval: int, eval_interval: int):
        # initialize master from file
        if saved_model is not None:
            self.load_file(saved_model)

        # initialize parameters
        epoch = 0
        # index_model = self.multi_select.initial_model()

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
            self.ret_list = []

            while not batch_ready:
                # get policy distribution
                act, act_log_prob, value_state = self.sample_action(obs)

                next_obs, rew, terminated, truncated, info = self.env.step(act)

                self.buff.add_exp(obs, act, act_log_prob, value_state, rew, next_obs, terminated, truncated)
                self.ret = self.ret * self.gamma + rew
                self.ret_list.append(self.ret)

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()
                    self.ret = 0

                # check if batch is done
                batch_ready = self.buff.buffer_full()

            all_states, all_next_states, all_actions_np, all_log_probs, all_values, all_rewards, all_term, \
                all_trunc = self.buff.get_clear_batch(self.batch_rng, self.batch_size)

            self.ret_rms.update(np.array(self.ret_list))
            all_n_rewards = all_rewards / np.sqrt(self.ret_rms.var + 1)
            # Train CLEAR
            pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min = \
                self.train(all_actions_np, all_log_probs, all_values, all_next_states, all_n_rewards, all_states,
                           all_term, all_trunc)

            norm_entropy = (real_entropy - self.minimum_entropy) / (self.maximum_entropy - self.minimum_entropy)
            self.tune_entropy(all_states)

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_a2c(self.buff, ret_np, v_np, adv_np, pg_loss, v_loss,
                        self.entropy_factor, norm_entropy, log_pi_min, elapsed, self.monitor, proj_eta, epoch, 0)
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

    def evaluate(self, epoch: int, skip_tb: bool):
        # initialize env
        # seems like gym objects have no seed
        # self.env.seed(self.seed)
        obs, _ = self.env_eval.reset()
        self.buff_eval.reset_head()
        self.buff_eval.len_roll = 0
        self.buff_eval.rew_roll = 0
        last_time = time.time()

        # training process
        while not self.buff_eval.buffer_full():
            # get policy distribution
            act, act_log_prob, value_state = self.sample_action(obs)

            next_obs, rew, terminated, truncated, info = self.env_eval.step(act)

            self.buff_eval.add_exp(obs, act, act_log_prob, value_state, rew, next_obs, terminated, truncated)

            # next state
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env_eval.reset()

        # check elapsed time
        curr_time = time.time()
        elapsed = curr_time - last_time

        if not skip_tb:
            log_stats_core(self.buff_eval, elapsed, self.monitor, epoch, eval_mode=True)
        print_eval(self.buff_eval, elapsed, epoch)
