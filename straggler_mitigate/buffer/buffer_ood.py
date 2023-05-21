from typing import Callable, List
import numpy as np


class OutOfDSampler(object):
    def __init__(self, obs_len: int, window: int, capacity: int, distant: Callable):
        self.cap = capacity
        self.win = window
        self.obs_len = obs_len
        self.is_distant = distant

        self.states_fifo = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.recent_states = np.zeros([self.win, self.obs_len], dtype=np.float32)

        # buffer head
        self.num_samples_so_far = 0
        self.i_fifo = 0
        self.i_win = 0

    def get(self, rng: np.random.RandomState, batch_size: int) -> List[np.ndarray]:
        if self.num_samples_so_far == 0:
            return []
        recents = self.recent_states[:min(self.num_samples_so_far, self.win)]
        alls = self.states_fifo[:min(self.num_samples_so_far, self.cap)]
        rets = []
        resamples = 0
        while len(rets) < batch_size and resamples < 5:
            new_rets = alls[rng.choice(len(alls), size=batch_size)]
            new_distant = self.is_distant(new_rets, recents)
            rets.extend(new_rets[new_distant])
            resamples += 1
        if len(rets) >= batch_size:
            return rets[:batch_size]
        else:
            return []

    def add_exp(self, state: np.ndarray):
        self.states_fifo[self.i_fifo] = state
        self.i_fifo = (self.i_fifo + 1) % self.cap
        self.recent_states[self.i_win] = state
        self.i_win = (self.i_win + 1) % self.win
        self.num_samples_so_far += 1

    def add_many_exp(self, states: np.ndarray):
        if self.i_fifo + len(states) <= self.cap:
            self.states_fifo[self.i_fifo: self.i_fifo+len(states)] = states
        else:
            self.states_fifo[self.i_fifo:] = states[:self.cap-self.i_fifo]
            self.states_fifo[:len(states)+self.i_fifo-self.cap] = states[self.cap-self.i_fifo:]
        self.i_fifo = (self.i_fifo + len(states)) % self.cap

        if self.i_win + len(states) <= self.win:
            self.recent_states[self.i_win: self.i_win+len(states)] = states
        else:
            self.recent_states[self.i_win:] = states[:self.win-self.i_win]
            self.recent_states[:len(states)+self.i_win-self.win] = states[self.win-self.i_win:]
        self.i_win = (self.i_win + len(states)) % self.win

        self.num_samples_so_far += len(states)

