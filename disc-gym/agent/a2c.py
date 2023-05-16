import time
from typing import List, Tuple
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_a2c, print_eval, log_stats_core
from agent.core_alg.core_pg import train_actor_critic, train_entropy
from neural_net.nn import FullyConnectNN, FCNPolicy
from buffer.buffer import TransitionBuffer
from utils.proj_time import ProjectFinishTime
from env.base import DiscGym
from utils.rms import RunningMeanStd


class TrainerNet(object):
    def __init__(self, environment: DiscGym, environment_eval: DiscGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float, lam: float,
                 auto_target_entropy: float, ent_lr: float, len_buff_eval: int):

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

        self.policy_net = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                    final_layer_act=False)
        self.policy_net = torch.jit.script(self.policy_net).to(self.device)
        self.value_net = FullyConnectNN(self.obs_len, nn_hids, 1, 1, act=nn.ReLU, final_layer_act=False)
        self.value_net = torch.jit.script(self.value_net).to(self.device)

        self.batch_size = batch_size
        self.buff = TransitionBuffer(self.obs_len, self.act_len, batch_size, reward_scale)
        self.buff_eval = TransitionBuffer(self.obs_len, self.act_len, len_buff_eval, reward_scale)

        self.entropy_factor = entropy_max
        self.entropy_max = entropy_max
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min

        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=lr_rate, weight_decay=1e-4, eps=1e-5)
        self.net_opt_p_lr = lr_rate
        self.net_opt_v = torch.optim.Adam(self.value_net.parameters(), lr=val_lr_rate, weight_decay=1e-4, eps=1e-5)

        self.it = 0

        self.maximum_entropy = self.act_len * np.log(self.act_bins)
        self.minimum_entropy = 0

        self.auto_target_entropy = auto_target_entropy
        if auto_target_entropy > 0:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_entropy = self.maximum_entropy * auto_target_entropy
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=ent_lr, weight_decay=1e-4)

        self.net_loss = torch.nn.MSELoss(reduction='mean')
        self.gamma = gamma
        self.lamb = lam

        self.ret = 0
        self.ret_list = []
        self.ret_rms = RunningMeanStd(shape=())

    def para_lr(self):
        self.monitor.add_scalar('Adam/PolLR', self.net_opt_p_lr, self.it)

    def load_file(self, path_or_dict: str or dict):
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

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        return self.policy_net.sample_action(torch.as_tensor(obs, dtype=torch.float, device=self.device)).numpy()

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                  np.ndarray, float]:
        self.para_lr()
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_actor_critic(self.value_net, self.policy_net, self.net_opt_p, self.net_opt_v, self.net_loss,
                               self.device, actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np,
                               self.gamma, self.lamb, self.entropy_factor, monitor=self.monitor, it=self.it)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min

    def tune_entropy(self, obs_np: np.ndarray):
        if self.auto_target_entropy <= 0:
            # entropy decay
            self.entropy_factor = max(self.entropy_factor - self.entropy_decay, self.entropy_min)
        else:
            ent_loss, self.entropy_factor = train_entropy(self.policy_net, obs_np, self.log_entropy, self.opt_ent,
                                                          self.device, self.target_entropy, self.entropy_max)

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
            batch_ready = False
            self.buff.reset_head()
            self.ret_list = []

            while not batch_ready:
                # get policy distribution
                act = self.sample_action(obs)

                next_obs, rew, terminated, truncated, info = self.env.step(act)

                self.buff.add_exp(obs, act, rew, next_obs, terminated, truncated)
                self.ret = self.ret * self.gamma + rew
                self.ret_list.append(self.ret)

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()
                    self.ret = 0

                # check if batch is done
                batch_ready = self.buff.buffer_full()

            all_states, all_next_states, all_actions_np, all_rewards, all_term, all_trunc = self.buff.get()

            self.ret_rms.update(np.array(self.ret_list))
            all_n_rewards = all_rewards / np.sqrt(self.ret_rms.var + 1e-8)
            # Train A2C
            pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min = \
                self.train(all_actions_np, all_next_states, all_n_rewards, all_states, all_term, all_trunc)

            norm_entropy = (real_entropy - self.minimum_entropy) / (self.maximum_entropy - self.minimum_entropy)
            self.tune_entropy(all_states)

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_a2c(self.buff, self.policy_net, ret_np, v_np, adv_np, pg_loss, v_loss,
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
        last_time = time.time()

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

        if not skip_tb:
            log_stats_core(self.buff_eval, elapsed, self.monitor, epoch, eval_mode=True)
        print_eval(self.buff_eval, elapsed, epoch)
