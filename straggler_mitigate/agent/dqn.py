import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_dqn
from agent.core_alg.core_dqn import sample_action, train_dqn, soft_copy
from agent.train_wheels import safe_condition
from buffer.buffer_sac import TransitionBuffer
from cenv.clb.pyenv import PyLoadBalanceEnv
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

        self.q_net = torch.jit.script(PermInvNet(self.obs_len, self.act_len, config.num_servers,
                                                 self.aux_state).to(self.device))
        self.target_net = torch.jit.script(PermInvNet(self.obs_len, self.act_len, config.num_servers,
                                                      self.aux_state).to(self.device))
        soft_copy(self.target_net, self.q_net, 1)

        self.buff = TransitionBuffer(self.obs_len, config.num_epochs * config.off_policy_learn_steps,
                                     config.off_policy_learn_steps)
        self.batch_rng = np.random.RandomState(seed=config.seed)

        self.eps = 1
        self.act_rng = np.random.RandomState(seed=config.seed)
        self.rand_countdown = config.off_policy_random_epochs * config.off_policy_learn_steps

        self.net_opt_q = torch.optim.Adam(self.q_net.parameters(), lr=config.lr_rate, weight_decay=1e-4)

        self.net_loss = torch.nn.MSELoss(reduction='mean')

        self.gamma_rate = np.log(config.cont_decay) / 1000

        arrival_scale, size_scale = self.env.get_scales()
        np.save(self.output_folder + 'scales.npy', {'size': size_scale, 'arrival': arrival_scale})

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

    def sample_action(self, obs: np.ndarray) -> int:
        if self.rand_countdown > 0:
            self.rand_countdown -= 1
            return self.act_rng.choice(self.act_len)
        elif self.act_rng.random() < self.eps:
            return self.act_rng.choice(self.act_len)
        else:
            return sample_action(self.q_net, obs, self.device)

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              times_np: np.ndarray, dones_np: np.ndarray) -> [float, np.ndarray, np.ndarray]:
        return train_dqn(actions_np, next_obs_np, rewards_np, obs_np, dones_np, self.q_net, self.target_net,
                         self.net_opt_q, self.net_loss, self.device, self.gamma_rate, next_obs_np[:, -1], self.max_act,
                         times_np)

    def tune_eps(self):
        if self.rand_countdown == 0:
            self.eps = max(self.eps - config.eps_decay, config.eps_min)

    def save_model(self, epoch, last=False):
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
        self.env.seed(config.seed)
        obs = np.copy(self.env.reset())

        # Hack, because info_orig doesn't exist before step
        info_orig = {'unsafe': False}

        # training process
        while epoch < config.num_epochs:
            # collect experience
            self.buff.reset_head()

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

                curr_wall_time = info_orig['curr_time']

                self.buff.add_exp(obs, unclipped_obs, act, rew, next_obs_orig, done, self.env.get_work_measure(),
                                  info_orig['time_elapsed'])
                self.buff.update_info(info_orig['rew_vec_orig'], info_orig['time_elapsed'], info_orig['server_time'])

                train_wheels_engaged_len += 1

                # next state
                obs = np.copy(next_obs_orig)

            all_states, all_next_states, all_actions_np, all_rewards, all_dones, time_buffer = \
                self.buff.get_batch(self.batch_rng, config.off_policy_batch_size)

            # Train DQN
            v_loss, q_val, trg_val = \
                self.train(all_actions_np, all_next_states, all_rewards, all_states, time_buffer, all_dones)

            self.tune_eps()

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not config.skip_tb:
                log_dqn(self.buff, q_val, trg_val, v_loss, self.eps, train_wheels_engaged_sum/train_wheels_engaged_len,
                        elapsed, start_wall_time, self.monitor, proj_eta, epoch, curr_wall_time,
                        self.env.timeline_len())
            else:
                proj_eta.update_progress(epoch)

            # save model
            if epoch % config.save_interval == 0:
                self.save_model(epoch)

            # update counter
            epoch += 1

        self.save_model(epoch, last=True)
