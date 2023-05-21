import time
from typing import Tuple
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_stats_sac
from agent.core_alg.core_pg import sample_action
from agent.core_alg.core_sac import train_soft_actor_critic, soft_copy, train_entropy
from agent.train_wheels import safe_condition
from buffer.buffer_mbcd import TransitionBuffer
from buffer.buffer_fifo import TransitionBuffer as TransitionBufferFiFo
from cenv.clb.pyenv import PyLoadBalanceEnv
from agent.core_alg.core_mbpo import MBPO
from param import config
from neural_net.nn_perm import PermInvNet
from utils.proj_time import ProjectFinishTime


class TrainerNet(object):
    def __init__(self, environment: PyLoadBalanceEnv, monitor: SummaryWriter, output_folder: str):
        self.device = torch.device(config.device)
        if config.device != "cpu":
            torch.cuda.set_device(self.device)
        # CUDA seed affects NN initial weights and policy network decisions
        torch.random.manual_seed(config.seed)

        self.obs_len = environment.get_observation_len()
        self.act_len = len(config.lb_timeout_levels)
        self.max_act = self.act_len - 1
        self.env = environment
        self.job_gen = environment.get_env_job_gen()
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

        self.buff_fifo = TransitionBufferFiFo(self.obs_len, config.off_policy_learn_steps)

        self.buff_len = config.num_epochs * config.off_policy_learn_steps
        self.buff = TransitionBuffer(self.obs_len, self.buff_len)
        self.batch_rng = np.random.RandomState(seed=config.seed)

        self.entropy_factor = config.entropy_max
        self.act_rng = np.random.RandomState(seed=config.seed)
        self.rand_countdown = config.off_policy_random_epochs * config.off_policy_learn_steps

        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr_rate, weight_decay=1e-4)
        self.net_opt_v_1 = torch.optim.Adam(self.critic_1_local.parameters(), lr=config.val_lr_rate,
                                            weight_decay=1e-4)
        self.net_opt_v_2 = torch.optim.Adam(self.critic_2_local.parameters(), lr=config.val_lr_rate,
                                            weight_decay=1e-4)

        if config.auto_target_entropy is not None:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_entropy = np.log(self.act_len) * config.auto_target_entropy
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=config.ent_lr, weight_decay=1e-4)

        self.net_loss = torch.nn.MSELoss(reduction='mean')

        self.gamma_rate = np.log(config.cont_decay) / 1000

        self.deep_mbpo = MBPO(state_dim=self.obs_len,
                              action_dim=self.act_len,
                              memory_capacity=100000)

        arrival_scale, size_scale = self.env.get_scales()
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

    def save_model(self, epoch):
        self.save_file(self.output_folder + '/models/model_{}'.format(epoch))

    def sample_many_actions(self, obs: np.ndarray) -> np.ndarray:
        pi_cpu = self.policy_net.sample_policy(torch.as_tensor(obs, dtype=torch.float, device=self.device))
        # array of shape (BATCH, act_dim)
        acts = (pi_cpu[:, :-1].cumsum(-1) <= torch.rand((obs.shape[0], 1))).sum(dim=-1).numpy()
        return acts

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
        self.env.seed(config.seed)
        obs = np.copy(self.env.reset())
        # Hack, because info_orig doesn't exist before step
        info_orig = {'unsafe': False}

        # training process
        while epoch < config.num_epochs:
            # collect experience
            self.buff_fifo.reset_head()

            for _ in range(config.off_policy_learn_steps):
                # observe queue sizes
                unclipped_obs = self.env.queue_sizes()

                if info_orig['unsafe']:
                    # Max Action
                    act = self.max_act
                    exaggeration = config.extra_multiply_penalty if safe_condition(obs) else 1
                    train_wheels_engaged_sum += 1
                else:
                    # get policy distribution
                    act = self.sample_action(obs)
                    exaggeration = 1

                next_obs_orig, rew, done, info_orig = self.env.step(act, 0)
                rew = -rew * exaggeration

                self.buff_fifo.add_exp(obs, unclipped_obs, act, rew, done, self.env.get_work_measure())
                self.buff_fifo.update_info(info_orig['rew_vec_orig'], info_orig['time_elapsed'],
                                           info_orig['server_time'])

                act_1hot = np.zeros(self.act_len)
                act_1hot[act] = 1

                self.deep_mbpo.add_experience(obs.copy(), act_1hot, rew / config.reward_scale, np.copy(next_obs_orig),
                                              done)
                self.buff.add_exp(obs, act, rew, next_obs_orig.copy(), done, info_orig['time_elapsed'])

                curr_wall_time = info_orig['curr_time']

                train_wheels_engaged_len += 1

                # next state
                obs = np.copy(next_obs_orig)

            self.deep_mbpo.train()

            all_states, _, _, _, all_dones, time_buffer = self.buff.get_batch(self.batch_rng,
                                                                              config.off_policy_batch_size)

            all_fake_acts = self.sample_many_actions(all_states)
            assert all_fake_acts.ndim == 1
            all_fake_acts = np.expand_dims(all_fake_acts, axis=-1)
            all_fake_next_states, all_fake_rewards = self.deep_mbpo.predict(all_states, all_fake_acts)

            # Train SAC
            pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2 = \
                self.train(all_fake_acts, all_fake_next_states, all_fake_rewards, all_states, time_buffer, all_dones)

            norm_entropy = real_entropy / - np.log(self.act_len)
            self.tune_entropy(all_states)

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            log_stats_sac(self.buff_fifo, q_target, q_local_1, q_local_2, pg_loss, v_loss_1, v_loss_2,
                          self.entropy_factor, norm_entropy, train_wheels_engaged_sum/train_wheels_engaged_len, elapsed,
                          start_wall_time, self.monitor, proj_eta, epoch, curr_wall_time, self.env.timeline_len())

            # save model
            if epoch % config.save_interval == 0:
                self.save_model(epoch)

            # update counter
            epoch += 1

        self.save_model(epoch)
