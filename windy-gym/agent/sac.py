import time
from typing import List, Tuple
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import print_eval, log_stats_core, log_sac
from agent.core_alg.core_dqn import soft_copy
from agent.core_alg.core_pg import train_entropy
from agent.core_alg.core_sac import train_sac
from neural_net.nn import FCNPolicy
from buffer.buffer import TransitionBuffer
from utils.proj_time import ProjectFinishTime
from env.base import WindyGym
from utils.rms import RunningMeanStd


class TrainerNet(object):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, off_policy_random_epochs: int, off_policy_learn_steps: int,
                 tau: float, num_epochs: int, len_buff_eval: int, parallel: bool):

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
                                    final_layer_act=False).to(self.device)
        self.critic_1_local = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                        final_layer_act=False).to(self.device)
        self.critic_2_local = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                        final_layer_act=False).to(self.device)
        self.critic_1_target = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                         final_layer_act=False).to(self.device)
        self.critic_2_target = FCNPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                         final_layer_act=False).to(self.device)

        self.policy_net = torch.jit.script(self.policy_net).to(self.device)
        self.critic_1_local = torch.jit.script(self.critic_1_local).to(self.device)
        self.critic_2_local = torch.jit.script(self.critic_2_local).to(self.device)
        self.critic_1_target = torch.jit.script(self.critic_1_target).to(self.device)
        self.critic_2_target = torch.jit.script(self.critic_2_target).to(self.device)

        soft_copy(self.critic_1_target, self.critic_1_local, 1)
        soft_copy(self.critic_2_target, self.critic_2_local, 1)

        self.batch_size = batch_size
        self.buff = TransitionBuffer(self.obs_len, self.act_len, num_epochs * off_policy_learn_steps, reward_scale,
                                     size_fifo=off_policy_learn_steps)
        self.batch_rng = np.random.RandomState(seed=self.seed)
        self.buff_eval = TransitionBuffer(self.obs_len, self.act_len, len_buff_eval, reward_scale)

        self.entropy_factor = entropy_max
        self.entropy_max = entropy_max
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min
        self.off_policy_random_epochs = off_policy_random_epochs
        self.act_rng = np.random.RandomState(seed=self.seed)
        self.rand_countdown = off_policy_random_epochs * off_policy_learn_steps

        self.tau = tau
        self.off_policy_learn_steps = off_policy_learn_steps

        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=lr_rate, weight_decay=1e-4, eps=1e-5)
        self.net_opt_p_lr = lr_rate
        self.net_opt_v_1 = torch.optim.Adam(self.critic_1_local.parameters(), lr=val_lr_rate, weight_decay=1e-4,
                                            eps=1e-5)
        self.net_opt_v_2 = torch.optim.Adam(self.critic_2_local.parameters(), lr=val_lr_rate, weight_decay=1e-4,
                                            eps=1e-5)

        self.maximum_entropy = self.act_len * np.log(self.act_bins)
        self.minimum_entropy = 0

        self.auto_target_entropy = auto_target_entropy
        if auto_target_entropy > 0:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_entropy = self.maximum_entropy * auto_target_entropy
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=ent_lr, weight_decay=1e-4)

        self.it = 0

        self.net_loss = torch.nn.MSELoss(reduction='mean')
        self.gamma = gamma

        self.ret_rms = RunningMeanStd(shape=())

        self.parallel = parallel

    def para_lr(self):
        self.monitor.add_scalar('Adam/PolLR', self.net_opt_p_lr, self.it)
        if self.parallel and self.it >= 5000 and self.it % 1000 == 0:
            self.net_opt_p_lr = 0.5 * self.net_opt_p_lr
            for param_group in self.net_opt_p.param_groups:
                param_group['lr'] = self.net_opt_p_lr

    def load_file(self, path_or_dict: str or dict):
        if isinstance(path_or_dict, str):
            state_dict = torch.load(path_or_dict, map_location=self.device)
        else:
            state_dict = path_or_dict
        self.policy_net.load_state_dict(state_dict[0])
        self.critic_1_local.load_state_dict(state_dict[1])
        self.critic_2_local.load_state_dict(state_dict[2])
        self.critic_1_target.load_state_dict(state_dict[3])
        self.critic_2_target.load_state_dict(state_dict[4])
        self.net_opt_p.load_state_dict(state_dict[5])
        self.net_opt_v_1.load_state_dict(state_dict[6])
        self.net_opt_v_2.load_state_dict(state_dict[7])

    def save_file(self, path: str):
        state_dict = [self.policy_net.state_dict(), self.critic_1_local.state_dict(), self.critic_2_local.state_dict(),
                      self.critic_1_target.state_dict(), self.critic_2_target.state_dict(), self.net_opt_p.state_dict(),
                      self.net_opt_v_1.state_dict(), self.net_opt_v_2.state_dict()]
        torch.save(state_dict, path)

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        if self.rand_countdown > 0:
            self.rand_countdown -= 1
            return self.act_rng.choice(self.act_bins, size=(self.act_len, ))
        else:
            pi_cpu = self.policy_net.pi(torch.as_tensor(obs, dtype=torch.float, device=self.device)).cpu()
            # array of shape (1, act_dim, act_bins)
            act = (pi_cpu[0, :, :-1].cumsum(-1) <= torch.rand((self.act_len, 1))).sum(dim=-1).numpy()
            return act

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              term_np: np.ndarray, trunc_np: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, np.ndarray,
                                                                  np.ndarray]:
        self.para_lr()
        pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2 = \
            train_sac(actions_np, next_obs_np, rewards_np, obs_np, term_np, trunc_np, self.policy_net,
                      self.critic_1_target, self.critic_2_target, self.critic_1_local, self.critic_2_local,
                      self.net_opt_v_1, self.net_opt_v_2, self.net_opt_p, self.net_loss, self.device, self.gamma,
                      self.entropy_factor, self.tau)
        self.it += 1
        return pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2

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
            self.buff.reset_head()

            for _ in range(self.off_policy_learn_steps):
                act = self.sample_action(obs)

                next_obs, rew, terminated, truncated, info = self.env.step(act)

                self.buff.add_exp(obs, act, rew, next_obs, terminated, truncated)

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()

            all_states, all_next_states, all_actions_np, all_rewards, all_term, all_trunc = \
                self.buff.get_batch(self.batch_rng, self.batch_size)

            all_n_rewards = all_rewards / np.sqrt(self.ret_rms.var + 1)

            # Train SAC
            pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2 = \
                self.train(all_actions_np, all_next_states, all_n_rewards, all_states, all_term, all_trunc)

            self.ret_rms.update(q_target)

            norm_entropy = (real_entropy - self.minimum_entropy) / (self.maximum_entropy - self.minimum_entropy)
            self.tune_entropy(all_states)

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_sac(self.buff, q_target, q_local_1, q_local_2, pg_loss, v_loss_1, v_loss_2, self.entropy_factor,
                        norm_entropy, elapsed, self.monitor, proj_eta, epoch)
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
        self.buff_eval.reset_episode()
        self.buff_eval.len_roll = 0
        self.buff_eval.rew_roll = 0
        last_time = time.time()

        pre_rc = self.rand_countdown

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

        self.rand_countdown = pre_rc

        if not skip_tb:
            log_stats_core(self.buff_eval, elapsed, self.monitor, epoch, eval_mode=True)
        print_eval(self.buff_eval, elapsed, epoch)
