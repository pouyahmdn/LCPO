from collections import deque
import gym
import numpy as np

from env.base import WindyGym


class WindyPendulum(WindyGym):
    def __init__(self, wind_arr: np.ndarray, threshold: float, bins: int = 9, parallel: bool = False,
                 eval_mode: bool = False):
        super(WindyPendulum, self).__init__(gym.make('Pendulum-v1'), wind_arr, bins)
        assert not (eval_mode and parallel)
        assert self.wind_arr.ndim == 2
        assert self.wind_arr.shape[1] == 2
        self.max_speed = self.env.max_speed
        self.max_torque = self.env.max_torque
        self.max_wind = 4
        assert np.all(np.abs(self.wind_arr)) < self.max_wind
        high = np.array([1.0, 1.0, self.max_speed] + 6*[self.max_wind], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.obs_dim = self.observation_space.shape[0]
        self.parallel = parallel
        self.eval_mode = eval_mode
        self.rng = np.random.default_rng(seed=1234)
        self.wind_index = 0
        self.eval_loop = deque(list(self.rng.choice(len(self.wind_arr)-200, 100)))
        self.thresh = threshold
        self.past_obs = None

    def _get_obs(self, orig_obs: np.ndarray):
        return np.array([orig_obs[0], orig_obs[1], orig_obs[2],
                         self.wind_arr[self.wind_index, 0],
                         self.wind_arr[self.wind_index-1, 0],
                         self.wind_arr[self.wind_index-2, 0],
                         self.wind_arr[self.wind_index, 1],
                         self.wind_arr[self.wind_index-1, 1],
                         self.wind_arr[self.wind_index-2, 1],
                         ],
                        dtype=np.float32)

    def step(self, action: np.ndarray):
        assert np.all(action < self.n_bins)
        assert np.all(action >= 0)
        act_cnt = np.take(self.act_bins, action)
        self.wind_index += 1
        if self.wind_index == len(self.wind_arr):
            self.wind_index = 0
        curr_wind = self.wind_arr[self.wind_index]
        # Apply wind as part of the torque
        act_n_wind = act_cnt + curr_wind[0] * self.past_obs[0] + curr_wind[1] * self.past_obs[1]
        self.past_obs, reward, terminated, truncated, info = self.env.step(act_n_wind)
        # Don't consider wind in reward
        act_cnt = np.clip(act_cnt, -self.max_torque, self.max_torque)[0]
        act_n_wind = np.clip(act_n_wind, -self.max_torque, self.max_torque)[0]
        reward += 0.001 * (act_n_wind ** 2 - act_cnt ** 2)
        return self._get_obs(self.past_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.parallel:
            self.wind_index = self.rng.choice(len(self.wind_arr))
        elif self.eval_mode:
            self.wind_index = self.eval_loop[0]
            self.eval_loop.rotate(-1)
        self.past_obs, other = self.env.reset(**kwargs)
        return self._get_obs(self.past_obs), other

    # def is_different(self, data: np.ndarray, base: np.ndarray) -> np.ndarray:
    #     mu, cov = np.mean(base[:, 3:9], axis=0), np.cov(base[:, 3:9], rowvar=False)
    #     cen_dat = data[:, 3:9] - mu
    #     ll = -(cen_dat @ np.linalg.inv(cov) * cen_dat).mean(axis=-1) / 2
    #     return ll < self.thresh

    def is_different(self, data: np.ndarray, base: np.ndarray) -> np.ndarray:
        assert base.shape[-1] == 9
        mu_base = np.r_[np.mean(base[:, 3:6]), np.mean(base[:, 6:9])]
        mu_data = data[:, 3:9].reshape((-1, 2, 3)).mean(axis=-1)
        dist = ((mu_data - mu_base)**2).sum(axis=-1)
        return dist > self.thresh
