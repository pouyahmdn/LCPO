import time
from typing import List, Tuple
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_dqn, print_eval, log_stats_core
from agent.core_alg.core_dqn import train_dqn, soft_copy
from neural_net.nn import FCNPolicy
from buffer.buffer import TransitionBuffer
from utils.proj_time import ProjectFinishTime
from env.base import DiscGym
from utils.rms import RunningMeanStd


class TrainerNet(object):
    def __init__(self, environment: DiscGym, environment_eval: DiscGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, eps_decay: float,
                 eps_min: float, num_epochs: int, lr_rate: float, gamma: float, off_policy_random_epochs: int,
                 off_policy_learn_steps: int, tau: float, len_buff_eval: int):

        self.device = torch.device(device)
        if device != "cpu":
            torch.cuda.set_device(self.device)
        # CUDA seed affects NN initial weights and policy network decisions
        self.seed = seed
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        self.obs_len = environment.obs_dim
        self.act_len = environment.action_dim
        self.act_bins = environment.n_bins
        self.env = environment
        self.env_eval = environment_eval
        self.monitor = monitor
        self.output_folder = output_folder

        self.q_net = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                               final_layer_act=False).to(self.device)
        self.q_net = torch.jit.script(self.q_net).to(self.device)
        self.target_net = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                    final_layer_act=False).to(self.device)
        self.target_net = torch.jit.script(self.target_net).to(self.device)
        soft_copy(self.target_net, self.q_net, 1)

        self.batch_size = batch_size
        self.buff = TransitionBuffer(self.obs_len, self.act_len, num_epochs * off_policy_learn_steps, reward_scale,
                                     size_fifo=off_policy_learn_steps)
        self.batch_rng = np.random.RandomState(seed=self.seed)
        self.buff_eval = TransitionBuffer(self.obs_len, self.act_len, len_buff_eval, reward_scale)

        self.eps = 1
        self.tau = tau
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.off_policy_random_epochs = off_policy_random_epochs
        self.off_policy_learn_steps = off_policy_learn_steps
        self.act_rng = np.random.RandomState(seed=self.seed)
        self.rand_countdown = off_policy_random_epochs * off_policy_learn_steps

        self.net_opt_q = torch.optim.Adam(self.q_net.parameters(), lr=lr_rate, weight_decay=1e-4, eps=1e-5)
        self.net_opt_q_lr = lr_rate

        self.it = 0

        self.net_loss = torch.nn.MSELoss(reduction='mean')
        self.gamma = gamma

        self.ret_rms = RunningMeanStd(shape=())

    def para_lr(self):
        self.monitor.add_scalar('Adam/PolLR', self.net_opt_q_lr, self.it)

    def load_file(self, path_or_dict: str or dict):
        if isinstance(path_or_dict, str):
            state_dict = torch.load(path_or_dict, map_location=self.device)
        else:
            state_dict = path_or_dict
        self.q_net.load_state_dict(state_dict[0])
        self.target_net.load_state_dict(state_dict[1])
        self.net_opt_q.load_state_dict(state_dict[2])

    def save_file(self, path: str):
        state_dict = [self.q_net.state_dict(), self.target_net.state_dict(), self.net_opt_q.state_dict()]
        torch.save(state_dict, path)

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        if self.rand_countdown > 0:
            self.rand_countdown -= 1
            return self.act_rng.choice(self.act_bins, size=(self.act_len, ))
        elif self.act_rng.random() < self.eps:
            return self.act_rng.choice(self.act_bins, size=(self.act_len, ))
        else:
            _, max_choice = self.q_net.max(torch.as_tensor(obs, dtype=torch.float, device=self.device))
            return max_choice[0].numpy()

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        self.para_lr()
        v_loss, q_val, trg_val = train_dqn(actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np, self.q_net,
                                           self.target_net, self.net_opt_q, self.net_loss, self.device, self.gamma,
                                           self.tau)
        self.it += 1
        return v_loss, q_val, trg_val

    def tune_eps(self):
        if self.rand_countdown == 0:
            self.eps = max(self.eps - self.eps_decay, self.eps_min)

    def save_model(self, epoch):
        self.save_file(self.output_folder + '/models/model_{}'.format(epoch))

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

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()

            assert self.buff.buffer_full()

            all_states, all_next_states, all_actions_np, all_rewards, all_term, all_trunc = \
                self.buff.get_batch(self.batch_rng, self.batch_size)

            all_n_rewards = all_rewards / np.sqrt(self.ret_rms.var + 1)

            # Train DQN
            v_loss, q_val, trg_val = self.train(all_actions_np, all_next_states, all_n_rewards, all_states, all_term,
                                                all_trunc)

            self.ret_rms.update(trg_val)

            self.tune_eps()

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_dqn(self.buff, q_val, trg_val, v_loss, self.eps, elapsed, self.monitor, proj_eta, epoch)
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
        last_time = time.time()

        pre_eps, pre_rc = self.eps, self.rand_countdown

        # training process
        while not self.buff_eval.buffer_full():
            # get policy distribution
            act = self.sample_action(obs)

            next_obs, rew, terminated, truncated, info = self.env_eval.step(act)

            self.buff_eval.add_exp(obs, act, rew, next_obs, terminated, truncated)

            # next state
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env_eval.reset()

        # check elapsed time
        curr_time = time.time()
        elapsed = curr_time - last_time

        self.eps, self.rand_countdown = pre_eps, pre_rc

        if not skip_tb:
            log_stats_core(self.buff_eval, elapsed, self.monitor, epoch, eval_mode=True)
        print_eval(self.buff_eval, elapsed, epoch)
