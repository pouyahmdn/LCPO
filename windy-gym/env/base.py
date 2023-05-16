import gym
from gym import Wrapper
import numpy as np


class WindyGym(Wrapper):
    def __init__(self, env: gym.Env, wind_arr: np.ndarray, bins=9):
        super(WindyGym, self).__init__(env)
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        self.wind_arr = wind_arr
        self.wind_index = 0
        self.n_bins = bins
        self.act_bins = np.array([np.linspace(self.env.action_space.low[i], self.env.action_space.high[i],
                                              bins, endpoint=True) for i in range(self.action_dim)])

    def step(self, action: np.ndarray):
        assert np.all(action < self.n_bins)
        assert np.all(action >= 0)
        act_continuous = np.take(self.act_bins, action)
        next_obs, reward, terminated, truncated, info = self.env.step(act_continuous)
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    @staticmethod
    def is_different(data: np.ndarray, base: np.ndarray) -> np.ndarray:
        return np.zeros(len(data), dtype=bool)
