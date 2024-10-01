from typing import Tuple, List
import numpy as np


class TransitionBuffer(object):
    def __init__(self, obs_len: int, act_len: int, size: int, reward_scale: float = 1, size_fifo: int = None):
        if size_fifo is None:
            size_fifo = size
        self.cap = size
        self.short_cap = size_fifo
        assert self.short_cap <= self.cap

        self.obs_len = obs_len
        self.act_len = act_len

        self.reward_scale = reward_scale

        self.states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.next_states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.actions_buffer = np.zeros([self.cap, self.act_len], dtype=np.int64)
        self.rewards_buffer = np.zeros([self.cap], dtype=np.float32)
        self.terminate_buffer = np.zeros([self.cap], dtype=bool)
        self.truncate_buffer = np.zeros([self.cap], dtype=bool)

        self.epi_counter = np.zeros([self.cap, 2], dtype=np.int64)

        self.states_fifo = np.zeros([self.short_cap, self.obs_len], dtype=np.float32)
        self.action_fifo = np.zeros([self.short_cap, self.act_len], dtype=np.int64)
        self.reward_fifo = np.zeros([self.short_cap], dtype=np.float32)
        self.terminate_fifo = np.zeros([self.short_cap], dtype=bool)
        self.truncate_fifo = np.zeros([self.cap], dtype=bool)
        self.episode_len_fifo = np.zeros([self.cap], dtype=int)
        self.episode_rew_fifo = np.zeros([self.cap], dtype=float)

        # buffer head
        self.num_samples_so_far = 0
        self.b = 0
        self.valid_samples = 0
        self.b_fifo = 0

        self.b_epi = 0
        self.b_epi_counter = 0

        self.rew_roll = 0
        self.len_roll = 0

    def __len__(self):
        return self.num_samples_so_far

    def reset_head(self):
        self.b_fifo = 0
        self.b_epi = 0

    def reset_episode(self):
        self.b_epi_counter = 0
        self.rew_roll = 0
        self.len_roll = 0
        self.epi_counter[0, 0] = 0

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.states_buffer[:self.valid_samples], \
            self.next_states_buffer[:self.valid_samples], \
            self.actions_buffer[:self.valid_samples], \
            self.rewards_buffer[:self.valid_samples] / self.reward_scale, \
            self.terminate_buffer[:self.valid_samples], \
            self.truncate_buffer[:self.valid_samples]

    def get_batch(self, rng: np.random.RandomState, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                              np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(self.valid_samples, batch_size)

        return self.states_buffer[indices], \
            self.next_states_buffer[indices], \
            self.actions_buffer[indices], \
            self.rewards_buffer[indices] / self.reward_scale, \
            self.terminate_buffer[indices], \
            self.truncate_buffer[indices]

    def get_batch_from_traj(self, rng: np.random.RandomState, min_batch_size: int) -> Tuple[List[np.ndarray],
                                                                                            List[np.ndarray],
                                                                                            List[np.ndarray],
                                                                                            List[np.ndarray],
                                                                                            List[np.ndarray],
                                                                                            List[np.ndarray]]:
        assert self.b_epi_counter > 0
        len_data = 0
        i_epi = []
        ret_data = [[], [], [], [], [], []]
        while len_data < min_batch_size:
            index = rng.choice(self.b_epi_counter)
            i_epi.append(index)
            len_data += self.epi_counter[index, 1] - self.epi_counter[index, 0]
            if self.epi_counter[index, 1] - self.epi_counter[index, 0] <= 0:
                import pdb
                pdb.set_trace()
            sl = slice(self.epi_counter[index, 0], self.epi_counter[index, 1])
            ret_data[0].append(self.states_buffer[sl])
            ret_data[1].append(self.next_states_buffer[sl])
            ret_data[2].append(self.actions_buffer[sl])
            ret_data[3].append(self.rewards_buffer[sl] / self.reward_scale)
            ret_data[4].append(self.terminate_buffer[sl])
            ret_data[5].append(self.truncate_buffer[sl])

        return ret_data[0], ret_data[1], ret_data[2], ret_data[3], ret_data[4], ret_data[5]

    def _place_exp(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminate: bool,
                   truncate: bool, index: int):
        self.states_buffer[index] = state
        self.next_states_buffer[index] = next_state
        self.actions_buffer[index] = action
        self.rewards_buffer[index] = reward
        self.terminate_buffer[index] = terminate
        self.truncate_buffer[index] = truncate

    def add_exp(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminate: bool,
                truncate: bool, **kwargs):
        self._place_exp(state, action, reward, next_state, terminate, truncate, self.b)
        self.b += 1
        self.valid_samples = min(1 + self.valid_samples, self.cap)
        self.states_fifo[self.b_fifo] = state
        self.action_fifo[self.b_fifo] = action
        self.reward_fifo[self.b_fifo] = reward
        self.terminate_fifo[self.b_fifo] = terminate
        self.truncate_fifo[self.b_fifo] = truncate
        self.rew_roll += reward
        self.len_roll += 1
        if terminate or truncate:
            # There is a chance that the trajectory rolls over back to the start, but we ignore this until we use
            # get_batch_from_traj()
            self.epi_counter[self.b_epi_counter, 1] = self.b
            self.episode_len_fifo[self.b_epi] = self.len_roll
            self.episode_rew_fifo[self.b_epi] = self.rew_roll
            self.len_roll = 0
            self.rew_roll = 0
            self.b_epi += 1
            self.b_epi_counter += 1
            if self.b_epi_counter == self.cap:
                self.b_epi_counter = 0
            if self.b != self.cap:
                self.epi_counter[self.b_epi_counter, 0] = self.b
            else:
                self.epi_counter[self.b_epi_counter, 0] = 0
        self.b_fifo += 1
        self.num_samples_so_far += 1
        if self.b == self.cap:
            self.b = 0

    def buffer_full(self) -> bool:
        return self.b_fifo >= self.short_cap
