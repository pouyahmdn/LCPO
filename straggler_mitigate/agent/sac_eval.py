import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.sac import TrainerNet as TrainerNetSAC
from agent.monitoring.core_log import log_stats_basic
from agent.train_wheels import safe_condition
from buffer.buffer_fifo import TransitionBuffer
from cenv.clb.pyenv import PyLoadBalanceEnv
from param import config
from utils.proj_time import ProjectFinishTime


class TrainerNet(TrainerNetSAC):
    def __init__(self, environment: PyLoadBalanceEnv, monitor: SummaryWriter, output_folder: str):
        super(TrainerNet, self).__init__(environment, monitor, output_folder)
        self.buff = TransitionBuffer(self.obs_len, config.master_batch)
        self.rand_countdown = 0

    def run_training(self):
        # initialize master from file
        self.load_file(config.saved_model)
        path_scales = "/".join(config.saved_model.rstrip('/').split('/')[:-2]) + "/scales.npy"
        scale_dict = np.load(path_scales, allow_pickle=True).item()
        self.env.set_scales(size_scale=scale_dict['size'], arrival_scale=scale_dict['arrival'])

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

                self.buff.add_exp(obs, unclipped_obs, act, rew, done, self.env.get_work_measure())
                self.buff.update_info(info_orig['rew_vec_orig'], info_orig['time_elapsed'], info_orig['server_time'])

                train_wheels_engaged_len += 1

                # next state
                obs = np.copy(next_obs_orig)

                # check if batch is done
                batch_ready = self.buff.buffer_full()

            # check elapsed time
            curr_time = time.time()
            elapsed = curr_time - last_time
            last_time = curr_time

            log_stats_basic(self.buff, elapsed, start_wall_time, self.monitor, proj_eta, epoch, curr_wall_time,
                            self.env.timeline_len(), train_wheels_engaged_sum/train_wheels_engaged_len)

            # update counter
            epoch += 1
