import time
from typing import List, Tuple
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_stats_sac
from agent.core_alg.core_pg import sample_action, train_entropy
from agent.core_alg.core_sac import train_soft_actor_critic, soft_copy, train_entropy
from agent.train_wheels import safe_condition
from buffer.buffer_sac import TransitionBuffer
from cenv.clb.pyenv import PyLoadBalanceEnv
from param import config
from neural_net.nn_perm import PermInvNet
from utils.proj_time import ProjectFinishTime


class TrainerNet(object):
    def __init__(self, environment_list: List[PyLoadBalanceEnv], monitor: SummaryWriter, output_folder: str):
        self.device = torch.device(config.device)
        if config.device != "cpu":
            torch.cuda.set_device(self.device)
        # CUDA seed affects NN initial weights and policy network decisions
        torch.random.manual_seed(config.seed)

        self.obs_len = environment_list[0].get_observation_len()
        self.act_len = len(config.lb_timeout_levels)
        self.max_act = self.act_len - 1
        self.env_s = environment_list
        self.monitor = monitor
        self.output_folder = output_folder

        np.random.seed(config.seed)

        self.aux_state = 12

        self.policy_net = torch.jit.script(PermInvNet(self.obs_len, self.act_len,
                                           config.num_servers, self.aux_state).to(self.device))
        self.critic_1_local = torch.jit.script(PermInvNet(self.obs_len, self.act_len,
                                               config.num_servers, self.aux_state).to(self.device))
        self.critic_2_local = torch.jit.script(PermInvNet(self.obs_len, self.act_len,
                                               config.num_servers, self.aux_state).to(self.device))
        self.critic_1_target = torch.jit.script(PermInvNet(self.obs_len, self.act_len,
                                                config.num_servers, self.aux_state).to(self.device))
        self.critic_2_target = torch.jit.script(PermInvNet(self.obs_len, self.act_len,
                                                config.num_servers, self.aux_state).to(self.device))
        # Copy local parameters to target parameters
        soft_copy(self.critic_1_target, self.critic_2_target, self.critic_1_local, self.critic_2_local, 1)

        self.buff = TransitionBuffer(self.obs_len, config.num_epochs*config.off_policy_learn_steps,
                                     config.off_policy_learn_steps*len(self.env_s))
        self.batch_rng = np.random.RandomState(seed=config.seed)

        self.entropy_factor = config.entropy_max
        self.act_rng = np.random.RandomState(seed=config.seed)
        self.rand_countdown = config.off_policy_random_epochs * config.off_policy_learn_steps

        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr_rate, weight_decay=1e-4)
        self.net_opt_v_1 = torch.optim.Adam(self.critic_1_local.parameters(), lr=config.val_lr_rate,
                                            weight_decay=1e-4)
        self.net_opt_v_2 = torch.optim.Adam(self.critic_2_local.parameters(), lr=config.val_lr_rate,
                                            weight_decay=1e-4)

        self.it = 0

        if config.auto_target_entropy is not None:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_entropy = np.log(self.act_len) * config.auto_target_entropy
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=config.ent_lr, weight_decay=1e-4)

        self.net_loss = torch.nn.MSELoss(reduction='mean')

        self.gamma_rate = np.log(config.cont_decay) / 1000

        scale_list = np.array([env.get_scales() for env in self.env_s])
        assert np.max(scale_list[:, 0]) - np.min(scale_list[:, 0]) < 1e-5, scale_list
        assert np.max(scale_list[:, 1]) - np.min(scale_list[:, 1]) < 1e-5, scale_list
        arrival_scale, size_scale = self.env_s[0].get_scales()
        np.save(self.output_folder + 'scales.npy', {'size': size_scale, 'arrival': arrival_scale})

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

    def sample_action(self, obs: np.ndarray) -> int:
        if self.rand_countdown > 0:
            self.rand_countdown -= 1
            return self.act_rng.choice(self.act_len)
        else:
            return sample_action(self.policy_net, obs, self.device)

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              times_np: np.ndarray, dones_np: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        return train_soft_actor_critic(actions_np, next_obs_np, rewards_np, obs_np, dones_np, self.policy_net,
                                       self.critic_1_target, self.critic_2_target, self.critic_1_local,
                                       self.critic_2_local, self.net_opt_v_1, self.net_opt_v_2, self.net_opt_p,
                                       self.net_loss, self.device, self.gamma_rate, self.entropy_factor, obs_np[:, -1],
                                       next_obs_np[:, -1], self.max_act, times_np)

    def tune_entropy(self, obs_np: np.ndarray):
        if config.auto_target_entropy is None:
            # entropy decay
            self.entropy_factor = max(self.entropy_factor - config.entropy_decay, config.entropy_min)
        else:
            ent_loss, self.entropy_factor = train_entropy(self.policy_net, obs_np, self.log_entropy, self.opt_ent,
                                                          self.device, self.target_entropy)

    def save_model(self, epoch: int):
        self.save_file(self.output_folder + '/models/model_{}'.format(epoch))

    def run_training(self):
        # initialize master from file
        if config.saved_model is not None:
            self.load_file(config.saved_model)

        # initialize parameters
        epoch = 0
        train_wheels_engaged_sum = 0
        train_wheels_engaged_len = 0

        # check elapsed time
        last_time = time.time()
        proj_eta = ProjectFinishTime(config.num_epochs)

        start_wall_time = 0
        curr_wall_time = 0

        # initialize env
        for env in self.env_s:
            env.seed(config.seed)
        obs_s = [np.copy(env.reset()) for env in self.env_s]

        # Hack, because info_orig doesn't exist before step
        info_orig_s = [{'unsafe': False} for _ in self.env_s]

        # training process
        while epoch < config.num_epochs:
            # collect experience
            self.buff.reset_head()

            n_es = 0

            for i in range(len(self.env_s)):
                cut = False
                valid_acts_this_env = 0
                while self.buff.samples_in_this_epoch != (i+1) * config.off_policy_learn_steps:
                    # observe queue sizes
                    unclipped_obs = self.env_s[i].queue_sizes()

                    if info_orig_s[i]['unsafe']:
                        # Max Action
                        act = self.max_act
                        exaggeration = config.extra_multiply_penalty if safe_condition(obs_s[i]) else 1
                        train_wheels_engaged_sum += 1
                    else:
                        # get policy distribution
                        act = self.sample_action(obs_s[i])
                        exaggeration = 1

                    valid_acts_this_env += 1

                    next_obs_orig, rew, done, info_orig_s[i] = self.env_s[i].step(act, 0)
                    rew = -rew * exaggeration

                    curr_wall_time = info_orig_s[i]['curr_time']

                    cut = valid_acts_this_env == config.off_policy_learn_steps

                    n_es += 1
                    self.buff.add_exp(obs_s[i], unclipped_obs, act, rew, next_obs_orig, done,
                                      self.env_s[i].get_work_measure(), info_orig_s[i]['time_elapsed'])
                    self.buff.update_info(info_orig_s[i]['rew_vec_orig'], info_orig_s[i]['time_elapsed'],
                                          info_orig_s[i]['server_time'])

                    train_wheels_engaged_len += 1
                    # next state
                    obs_s[i] = np.copy(next_obs_orig)

                assert cut is True

            assert self.buff.buffer_full()

            all_states, all_next_states, all_actions_np, all_rewards, all_dones, time_buffer = \
                self.buff.get_batch(self.batch_rng, config.off_policy_batch_size)

            # Train SAC
            pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2 = \
                self.train(all_actions_np, all_next_states, all_rewards, all_states, time_buffer, all_dones)

            norm_entropy = real_entropy / - np.log(self.act_len)
            self.tune_entropy(all_states)

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not config.skip_tb:
                log_stats_sac(self.buff, q_target, q_local_1, q_local_2, pg_loss, v_loss_1, v_loss_2,
                              self.entropy_factor, norm_entropy, train_wheels_engaged_sum / train_wheels_engaged_len,
                              elapsed, start_wall_time, self.monitor, proj_eta, epoch, curr_wall_time,
                              self.env_s[0].timeline_len())
            else:
                proj_eta.update_progress(epoch)

            # save model
            if epoch % config.save_interval == 0:
                self.save_model(epoch)

            # update counter
            epoch += 1

        self.save_model(epoch)
