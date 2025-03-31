from collections import deque
import gymnasium as gym
import numpy as np


class ShakyGrid(object):
    def __init__(self, eval_mode: bool = False):
        high = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.obs_dim = self.observation_space.shape[0]
        self.action_space = gym.spaces.Discrete(4)
        self.action_dim = 1
        self.n_bins = 4

        self.eval_mode = eval_mode
        self.eval_loop = deque([0])
        self.rng = np.random.default_rng(seed=1234)
        self.tr_index = 0

        # Trap
        self.tr_arr = np.r_[
                np.array([[0, 0]]).repeat(40000, axis=0),
                np.array([[0, 1]]).repeat(120000, axis=0),
                np.array([[0, 0]]).repeat(40000, axis=0),
        ]

        #  ################
        #  #    #    # E  #
        #  ################
        #  #    # PT # XX #
        #  ################
        #  #    #    # S  #
        #  ################

        # #: Cell lining
        # X: Wall
        # E: Goal
        # S: Start
        # PT: Possible Trap

        # row, col, from top left
        self.state = [2, 2]
        self.timer = 0
        self.triggered = [False, False]

        self.base_rew = -1
        self.trap_rew = -10
        self.trunc_len = 20

    def _get_obs(self):
        return np.array([
                         self.state[0],
                         self.state[1],
                         self.tr_arr[self.tr_index, 0],
                         self.tr_arr[self.tr_index, 1],
                         ],
                        dtype=np.float32)

    def step(self, action: np.ndarray):
        assert np.all(action < 4)
        assert np.all(action >= 0)
        direction = action[0]
        if direction == 0:          # Left
            self.state[1] = max(0, self.state[1]-1)
        elif direction == 2:          # Right
            self.state[1] = min(2, self.state[1]+1)
        elif direction == 1:          # Up
            if not (self.state[0] == 2 and self.state[1] == 2):
                self.state[0] = max(0, self.state[0]-1)
        elif direction == 3:          # Bottom
            if not (self.state[0] == 0 and self.state[1] == 2):
                self.state[0] = min(2, self.state[0]+1)
        else:
            raise ValueError

        if self.state[0] == 0 and self.state[1] == 2:
            reward = 0
        else:
            reward = self.base_rew
        curr_trace = self.tr_arr[self.tr_index]
        if self.state[0] == 1 and self.state[1] == 1 and curr_trace[1] == 1 and self.triggered[1] is False:
            reward += self.trap_rew
            self.triggered[1] = True
        self.timer += 1

        terminated = False
        truncated = self.timer == self.trunc_len

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self):
        if self.eval_mode:
            self.tr_index = self.eval_loop[0]
            self.eval_loop.rotate(-1)
        else:
            self.tr_index += 1
            if self.tr_index == len(self.tr_arr):
                self.tr_index = 0
        self.state = [2, 2]
        self.timer = 0
        self.triggered = [False, False]
        return self._get_obs(), {}

    def close(self):
        pass

    def is_different(self, data: np.ndarray, base: np.ndarray) -> np.ndarray:
        assert base.shape[-1] == 4
        mu_base = np.mean(base[:, 2:4], axis=0)
        mu_data = data[:, 2:4]
        dist = ((mu_data - mu_base)**2).sum(axis=-1)
        return dist > 0.5
