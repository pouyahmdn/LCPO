from typing import Tuple
import numpy as np

from param import config


class TransitionBuffer(object):
    def __init__(self, obs_len: int, size: int):
        self.cap = size
        self.obs_len = obs_len

        self.states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.next_states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.actions_buffer = np.zeros([self.cap, 1], dtype=np.int64)
        self.times_buffer = np.zeros([self.cap], dtype=np.float32)
        self.rewards_buffer = np.zeros([self.cap], dtype=np.float32)
        self.dones_buffer = np.zeros([self.cap], dtype=np.bool)

        # buffer head
        self.num_samples_so_far = 0
        self.b = 0

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

    def get_fifo(self, rng: np.random.RandomState, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                              np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(min(self.num_samples_so_far, self.cap), batch_size)

        return self.states_buffer[indices], \
            self.next_states_buffer[indices], \
            self.actions_buffer[indices], \
            self.rewards_buffer[indices] / config.reward_scale, \
            self.dones_buffer[indices], \
            self.times_buffer[indices]

    def _place_exp(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
                   time_job: float, index: int):
        self.states_buffer[index, :] = state
        self.next_states_buffer[index, :] = next_state
        self.actions_buffer[index, 0] = action
        self.rewards_buffer[index] = reward
        self.dones_buffer[index] = done
        self.times_buffer[index] = time_job

    def add_exp(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
                time_job: float):
        self._place_exp(state, action, reward, next_state, done, time_job, self.b)
        self.b += 1
        self.num_samples_so_far += 1
        if self.b == self.cap:
            self.b = 0
