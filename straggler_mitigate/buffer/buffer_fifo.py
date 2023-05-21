from typing import List, Tuple
import numpy as np

from param import config


class TransitionBuffer(object):
    def __init__(self, obs_len: int, fifo_size: int):
        self.cap = fifo_size
        self.obs_len = obs_len

        self.states_fifo = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.action_timeout_fifo = np.zeros([self.cap], dtype=np.float32)
        self.reward_fifo = np.zeros([self.cap], dtype=np.float32)
        self.dones_fifo = np.zeros([self.cap], dtype=np.bool)
        self.workload_fifo = np.zeros([self.cap], dtype=np.float32)
        self.state_queue_size_fifo = np.zeros([self.cap, config.num_servers], dtype=np.float32)

        # buffer head
        self.num_samples_so_far = 0
        self.samples_left_to_epoch = self.cap
        self.jct = []
        self.server_load = np.zeros(config.num_servers)
        self.time_elapsed = 0
        self.episode_len = 0

    def reset_head(self):
        self.samples_left_to_epoch = self.cap
        self.jct.clear()
        self.server_load = np.zeros(config.num_servers)
        self.time_elapsed = 0
        self.episode_len = 0

    def add_exp(self, state: np.ndarray, unclipped_state: np.ndarray, action: int, reward: float,
                done: bool, workload: float):
        self.state_queue_size_fifo[self.cap - self.samples_left_to_epoch, :] = unclipped_state
        self.action_timeout_fifo[self.cap - self.samples_left_to_epoch] = action
        self.workload_fifo[self.cap - self.samples_left_to_epoch] = workload
        self.reward_fifo[self.cap - self.samples_left_to_epoch] = reward
        self.dones_fifo[self.cap - self.samples_left_to_epoch] = done
        self.states_fifo[self.cap - self.samples_left_to_epoch, :] = state
        self.samples_left_to_epoch -= 1
        self.num_samples_so_far += 1
        self.episode_len = min(self.episode_len+1, self.cap)

    def buffer_full(self) -> bool:
        return self.samples_left_to_epoch <= 0

    def update_info(self, jct: List[float] or np.ndarray, t_elapsed: float, t_s_elapsed: float):
        self.jct.extend(jct)
        self.server_load = (self.server_load * self.time_elapsed + np.array(t_s_elapsed)) / (self.time_elapsed +
                                                                                             t_elapsed)
        self.time_elapsed += t_elapsed

    def get_server_load(self) -> Tuple[float, float, float]:
        avg_server_load = np.mean(self.server_load).item()
        max_server_load = np.max(self.server_load).item()
        min_server_load = np.min(self.server_load).item()
        return min_server_load, avg_server_load, max_server_load

    def get_job_completion_times(self) -> List[float]:
        return self.jct
