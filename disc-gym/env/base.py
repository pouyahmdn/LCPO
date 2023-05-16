import gym
from gym import Wrapper
import numpy as np


class DiscGym(Wrapper):
    def __init__(self, env: gym.Env, bins: int = 15):
        super(DiscGym, self).__init__(env)
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
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