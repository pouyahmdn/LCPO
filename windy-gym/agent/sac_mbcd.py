import copy
import time
from typing import List
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.monitoring.core_log import log_stats_basic, log_sac_mbcd
from agent.sac import TrainerNet as TrainerNetSAC
from buffer.buffer import TransitionBuffer
from agent.core_alg.core_mbcd import MBCD
from env.base import WindyGym
from utils.proj_time import ProjectFinishTime
from utils.rms import RunningMeanStd


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: WindyGym, environment_eval: WindyGym, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float,
                 auto_target_entropy: float, ent_lr: float, off_policy_random_epochs: int, off_policy_learn_steps: int,
                 tau: float, num_epochs: int, len_buff_eval: int, cusum: float, parallel: bool):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, auto_target_entropy, ent_lr, off_policy_random_epochs,
                                         off_policy_learn_steps, tau, num_epochs, len_buff_eval, parallel)

        self.ent_lr = ent_lr
        self.reward_scale = reward_scale

        self.buff_len = num_epochs * off_policy_learn_steps
        self.buff_s = {
            0: TransitionBuffer(self.obs_len, self.act_len, self.buff_len, reward_scale,
                                size_fifo=off_policy_learn_steps)
        }

        self.buff = TransitionBuffer(self.obs_len, self.act_len, off_policy_learn_steps, reward_scale)
        self.buff_eval = None

        self.explored = [0]
        self.prev_exp = {}
        self.prev_state_dicts = {}
        self.prev_rms = {}

        self.deep_mbcd = MBCD(state_dim=self.obs_len,
                              action_dim=self.act_len * self.act_bins,
                              memory_capacity=100000,
                              cusum_threshold=cusum,
                              max_std=0.5*6,
                              num_stds=2)
        self.model_train_freq = 250

    def get_state_dicts(self) -> List:
        return copy.deepcopy([self.policy_net.state_dict(),
                              self.critic_1_local.state_dict(),
                              self.critic_2_local.state_dict(),
                              self.critic_1_target.state_dict(),
                              self.critic_2_target.state_dict(),
                              self.net_opt_p.state_dict(),
                              self.net_opt_v_1.state_dict(),
                              self.net_opt_v_2.state_dict()])

    def get_exps(self) -> List:
        ret = copy.deepcopy([self.entropy_factor, self.rand_countdown])
        if self.auto_target_entropy > 0:
            ret.extend(copy.deepcopy([self.log_entropy, self.opt_ent]))
        return ret

    def apply_exps(self, past_exp: List):
        self.entropy_factor, self.rand_countdown = past_exp[:2]
        if self.auto_target_entropy > 0:
            self.log_entropy, self.opt_ent = past_exp[2:]

    def start_new_model(self, index_new: int):
        assert index_new not in self.prev_exp
        assert index_new not in self.prev_state_dicts
        self.entropy_factor = self.entropy_max
        self.rand_countdown = self.off_policy_random_epochs * self.off_policy_learn_steps
        if self.auto_target_entropy:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=self.ent_lr, weight_decay=1e-4)
        self.explored.append(index_new)
        self.buff_s[index_new] = TransitionBuffer(self.obs_len, self.act_len,
                                                  self.buff_len,
                                                  self.reward_scale,
                                                  size_fifo=self.off_policy_learn_steps)
        self.ret_rms = RunningMeanStd(shape=())

    def run_training(self, saved_model: str, num_epochs: int, skip_tb: bool, save_interval: int, eval_interval: int):
        # initialize master from file
        if saved_model is not None:
            self.load_file(saved_model)

        # initialize parameters
        epoch = 0
        index_model = 0

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
                changed = self.deep_mbcd.update_metrics(obs.copy(), act_1hot, rew, next_obs.copy(), terminated)

                self.monitor.add_scalar('MBCD/S', self.deep_mbcd.S[-1], epoch)
                self.monitor.add_scalar('MBCD/var_0', self.deep_mbcd.var_mean[0], epoch)
                self.monitor.add_scalar('MBCD/lp_0', self.deep_mbcd.log_prob[0], epoch)
                self.monitor.add_scalar('MBCD/nmlogpdf', self.deep_mbcd.new_model_log_pdf, epoch)
                self.monitor.add_scalar('MBCD/log_ratio', self.deep_mbcd.new_model_log_pdf-self.deep_mbcd.log_prob[0],
                                        epoch)
                self.monitor.add_scalar('MBCD/model_index', self.deep_mbcd.current_model, epoch)
                self.monitor.add_scalar('MBCD/num_models', len(self.deep_mbcd.models), epoch)

                self.deep_mbcd.add_experience(obs.copy(), act_1hot, rew, next_obs.copy(), terminated)
                if self.deep_mbcd.counter < 250:
                    self.model_train_freq = 10
                elif self.deep_mbcd.counter < 5000:
                    self.model_train_freq = 100
                elif self.deep_mbcd.counter < 40000:
                    self.model_train_freq = 250
                elif self.deep_mbcd.counter < 60000:
                    self.model_train_freq = 5000
                else:
                    self.model_train_freq = 2000

                if (changed and self.deep_mbcd.counter > 10) or (self.deep_mbcd.counter % self.model_train_freq == 0):
                    self.deep_mbcd.train()

                if changed:
                    # Log previous model
                    self.prev_state_dicts[index_model] = self.get_state_dicts()
                    self.prev_exp[index_model] = self.get_exps()
                    self.prev_rms[index_model] = self.ret_rms

                    # Start or resume
                    index_model = self.deep_mbcd.current_model
                    if index_model not in self.explored:
                        self.start_new_model(index_model)
                    else:
                        assert index_model in self.prev_exp
                        assert index_model in self.prev_state_dicts
                        self.load_file(self.prev_state_dicts[index_model])
                        self.apply_exps(self.prev_exp[index_model])
                        self.ret_rms = self.prev_rms[index_model]

                self.monitor.add_scalar('Multi/index_model', index_model, epoch)

                self.buff_s[index_model].add_exp(obs, act, rew, next_obs, terminated, truncated)

                if self.buff_s[index_model].buffer_full():

                    all_states, all_next_states, all_actions_np, all_rewards, all_term, all_trunc = \
                        self.buff_s[index_model].get_batch(self.batch_rng, self.batch_size)

                    all_n_rewards = all_rewards / np.sqrt(self.ret_rms.var + 1e-8)

                    # Train SAC
                    pg_loss, v_loss_1, v_loss_2, real_entropy, q_target, q_local_1, q_local_2 = \
                        self.train(all_actions_np, all_next_states, all_n_rewards, all_states, all_term, all_trunc)

                    self.ret_rms.update(q_target)

                    norm_entropy = (real_entropy - self.minimum_entropy) / (self.maximum_entropy - self.minimum_entropy)
                    self.tune_entropy(all_states)

                    if not skip_tb:
                        log_sac_mbcd(q_target, q_local_1, q_local_2, pg_loss, v_loss_1, v_loss_2, self.entropy_factor,
                                     norm_entropy, self.monitor, index_model, epoch)

                    self.tune_entropy(all_states)
                    self.buff_s[index_model].reset_head()

                # next state
                obs = next_obs

                if terminated or truncated:
                    obs, info = self.env.reset()

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            if not skip_tb:
                log_stats_basic(self.buff, elapsed, self.monitor, proj_eta, epoch)
            else:
                proj_eta.update_progress(epoch)

            # save model
            if epoch % save_interval == 0:
                self.save_model(epoch)

            # update counter
            epoch += 1

        self.save_model(epoch)
