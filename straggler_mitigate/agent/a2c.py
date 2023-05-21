import time
from typing import Tuple
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_a2c
from agent.core_alg.core_pg import sample_action, train_actor_critic, train_entropy
from agent.train_wheels import safe_condition
from cenv.clb.pyenv import PyLoadBalanceEnv
from param import config
from neural_net.nn_perm import PermInvNet
from buffer.buffer import TransitionBuffer
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
        self.value_net = torch.jit.script(PermInvNet(self.obs_len, 1,
                                          config.num_servers, self.aux_state).to(self.device))

        self.buff = TransitionBuffer(self.obs_len)

        self.entropy_factor = config.entropy_max

        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr_rate, weight_decay=1e-4)
        self.net_opt_v = torch.optim.Adam(self.value_net.parameters(), lr=config.val_lr_rate, weight_decay=1e-4)

        self.it = 0

        if config.auto_target_entropy is not None:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_entropy = np.log(self.act_len) * config.auto_target_entropy
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=config.ent_lr, weight_decay=1e-4)

        self.net_loss = torch.nn.MSELoss(reduction='mean')

        self.gamma_rate = np.log(config.cont_decay) / 1000

        arrival_scale, size_scale = self.env.get_scales()
        np.save(self.output_folder + 'scales.npy', {'size': size_scale, 'arrival': arrival_scale})

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

    def sample_action(self, obs: np.ndarray) -> int:
        return sample_action(self.policy_net, obs, self.device)

    def train(self, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              times_np: np.ndarray, dones_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                   np.ndarray, float]:
        pg_loss, v_loss, real_entropy, ret_np, v_np, log_pi_min, adv_np = \
            train_actor_critic(self.value_net, self.policy_net, self.net_opt_p, self.net_opt_v, self.net_loss,
                               self.device, actions_np, next_obs_np, rewards_np, obs_np, dones_np, obs_np[:, [-1]],
                               self.gamma_rate, self.entropy_factor, times_np=times_np, monitor=self.monitor,
                               it=self.it)
        self.it += 1
        return pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min

    def tune_entropy(self, obs_np):
        if config.auto_target_entropy is None:
            # entropy decay
            self.entropy_factor = max(self.entropy_factor - config.entropy_decay, config.entropy_min)
        else:
            ent_loss, self.entropy_factor = train_entropy(self.policy_net, obs_np, self.log_entropy, self.opt_ent,
                                                          self.device, self.target_entropy)

    def save_model(self, epoch):
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
            batch_ready = False
            self.buff.reset_head()

            while not batch_ready:
                # observe queue sizes
                unclipped_obs = self.env.queue_sizes()

                if info_orig['unsafe']:
                    # Max Action
                    tw_exp = True
                    act = self.max_act
                    exaggeration = config.extra_multiply_penalty if safe_condition(obs) else 1
                    train_wheels_engaged_sum += 1
                else:
                    # get policy distribution
                    tw_exp = False
                    act = self.sample_action(obs)
                    exaggeration = 1

                next_obs_orig, rew, done, info_orig = self.env.step(act, 0)
                rew = -rew * exaggeration

                curr_wall_time = info_orig['curr_time']

                self.buff.add_exp(obs, unclipped_obs, act, rew, next_obs_orig, done, self.env.get_work_measure(),
                                  info_orig['time_elapsed'], tw_exp=tw_exp)
                self.buff.update_info(info_orig['rew_vec_orig'], info_orig['time_elapsed'], info_orig['server_time'])

                train_wheels_engaged_len += 1
                # next state
                obs = np.copy(next_obs_orig)
                # check if batch is done
                batch_ready = self.buff.buffer_full()

            all_states, all_next_states, all_actions_np, all_rewards, all_dones, time_buffer = self.buff.get()

            # Train DQN
            pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np, log_pi_min = \
                self.train(all_actions_np, all_next_states, all_rewards, all_states, time_buffer, all_dones)

            norm_entropy = real_entropy / - np.log(self.act_len)
            self.tune_entropy(all_states)

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not config.skip_tb:
                log_a2c(self.buff, ret_np, v_np, adv_np, pg_loss, v_loss, self.entropy_factor, norm_entropy, log_pi_min,
                        train_wheels_engaged_sum/train_wheels_engaged_len, elapsed, start_wall_time, self.monitor,
                        proj_eta, epoch, curr_wall_time, self.env.timeline_len(), 0)
            else:
                proj_eta.update_progress(epoch)

            # save model
            if epoch % config.save_interval == 0:
                self.save_model(epoch)

            # update counter
            epoch += 1

        self.save_model(epoch)
