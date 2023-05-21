from typing import Tuple, List
import numpy as np

from param import config


class TransitionBuffer(object):
    def __init__(self, obs_len: int, size: int, size_fifo: int):
        self.cap = size
        self.short_cap = size_fifo
        self.obs_len = obs_len
        self.episode_len = self.short_cap

        self.states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.next_states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.actions_buffer = np.zeros([self.cap, 1], dtype=np.int64)
        self.times_buffer = np.zeros([self.cap], dtype=np.float32)
        self.rewards_buffer = np.zeros([self.cap], dtype=np.float32)
        self.dones_buffer = np.zeros([self.cap], dtype=np.bool)

        self.workload_fifo = np.zeros([self.short_cap], dtype=np.float32)
        self.dones_fifo = np.zeros([self.short_cap], dtype=np.bool)
        self.state_queue_size_fifo = np.zeros([self.short_cap, config.num_servers], dtype=np.float32)
        self.states_fifo = np.zeros([self.short_cap, self.obs_len], dtype=np.float32)
        self.action_timeout_fifo = np.zeros([self.short_cap], dtype=np.float32)
        self.reward_fifo = np.zeros([self.short_cap], dtype=np.float32)

        # buffer head
        self.num_samples_so_far = 0
        self.samples_left_to_epoch = self.short_cap
        self.b = 0
        self.jct = []
        self.server_load = np.zeros(config.num_servers)
        self.time_elapsed = 0

    @property
    def samples_in_this_epoch(self):
        return self.short_cap-self.samples_left_to_epoch

    def reset_head(self):
        self.samples_left_to_epoch = self.short_cap
        self.jct.clear()
        self.server_load = np.zeros(config.num_servers)
        self.time_elapsed = 0

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.states_buffer, \
            self.next_states_buffer, \
            self.actions_buffer, \
            self.rewards_buffer / config.reward_scale, \
            self.dones_buffer, \
            self.times_buffer

    def get_batch(self, rng: np.random.RandomState, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                              np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(min(self.num_samples_so_far, self.cap), batch_size)

        return self.states_buffer[indices], \
            self.next_states_buffer[indices], \
            self.actions_buffer[indices], \
            self.rewards_buffer[indices] / config.reward_scale, \
            self.dones_buffer[indices], \
            self.times_buffer[indices]

    def _place_exp(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, time_job: float, index: int):
        self.states_buffer[index, :] = state
        self.next_states_buffer[index, :] = next_state
        self.actions_buffer[index, 0] = action
        self.rewards_buffer[index] = reward
        self.dones_buffer[index] = done
        self.times_buffer[index] = time_job

    def add_exp(self, state: np.ndarray, unclipped_state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool, workload: float, time_job: float):
        if self.samples_left_to_epoch > 0:
            self.state_queue_size_fifo[self.short_cap - self.samples_left_to_epoch, :] = unclipped_state
            self.action_timeout_fifo[self.short_cap - self.samples_left_to_epoch] = action
            self.workload_fifo[self.short_cap - self.samples_left_to_epoch] = workload
            self.dones_fifo[self.short_cap - self.samples_left_to_epoch] = done
            self.states_fifo[self.short_cap - self.samples_left_to_epoch, :] = state
            self.reward_fifo[self.short_cap - self.samples_left_to_epoch] = reward
            self.samples_left_to_epoch -= 1
        else:
            print('WARNING: fifo buffer is full')
        self._place_exp(state, action, reward, next_state, done, time_job, self.b)
        self.b += 1
        self.num_samples_so_far += 1
        if self.b == self.cap:
            self.b = 0

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
