from collections import deque
import gymnasium as gym
import numpy as np
import pathlib
from env.base import WindyGym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.linalg import solve_triangular


# Adapted from
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/inverted_double_pendulum_v4.py
class MujocoWindyInvertedDoublePendulum(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent / "assets" / "windy_inv_double_pendulum.xml"),
            5,
            observation_space=observation_space,
            default_camera_config={
                "trackbodyid": 0,
                "distance": 4.1225,
                "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
            },
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.time_steps = 0
        self.max_time_steps = 200

    def step(self, action):
        self.time_steps += 1
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        terminated = bool(y <= 1)
        return ob, r, terminated, self.time_steps >= self.max_time_steps, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        self.time_steps = 0
        return self._get_obs()


class WindyDoubleInvertedPendulum(WindyGym):
    rescale_dict = {
        0: 8,
        1: 8,
        2: 8,
        3: 8,
    }

    def __init__(self, wind_arr: np.ndarray, threshold: float, bins: int = 9, parallel: bool = False,
                 eval_mode: bool = False, dist_func_type: str = 'l2'):
        super(WindyDoubleInvertedPendulum, self).__init__(MujocoWindyInvertedDoublePendulum(), wind_arr, bins)
        assert not (eval_mode and parallel)
        assert self.wind_arr.ndim == 2
        assert self.wind_arr.shape[1] == 2
        self.max_wind = 4
        assert np.all(np.abs(self.wind_arr)) < self.max_wind
        high = np.array([np.inf] * 11 + [self.max_wind] * 6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.obs_dim = self.observation_space.shape[0]
        self.parallel = parallel
        self.eval_mode = eval_mode
        self.rng = np.random.default_rng(seed=1234)
        self.wind_index = 0
        self.eval_loop = deque(list(self.rng.choice(len(self.wind_arr) - 200, 100)))
        self.thresh = threshold
        self.past_obs = None
        self.dist_func_type = dist_func_type

    def _get_obs(self, orig_obs: np.ndarray):
        return np.array([orig_obs[0], orig_obs[1], orig_obs[2], orig_obs[3],
                         orig_obs[4], orig_obs[5], orig_obs[6], orig_obs[7],
                         orig_obs[8], orig_obs[9], orig_obs[10],
                         self.wind_arr[self.wind_index, 0],
                         self.wind_arr[self.wind_index - 1, 0],
                         self.wind_arr[self.wind_index - 2, 0],
                         self.wind_arr[self.wind_index, 1],
                         self.wind_arr[self.wind_index - 1, 1],
                         self.wind_arr[self.wind_index - 2, 1],
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
        # Vertical angle, negative on the left, positive on right
        # Second angle is with respect to the first joint
        act_n_wind = act_cnt + curr_wind[0]
        joint_0_torque = curr_wind[0] * self.past_obs[3] + curr_wind[1] * self.past_obs[1]
        ang_j1 = self.env.data.qpos[1] + self.env.data.qpos[2]
        joint_1_torque = curr_wind[0] * np.cos(ang_j1) + curr_wind[1] * np.sin(ang_j1)
        self.past_obs, reward, terminated, truncated, info = self.env.step([act_n_wind, joint_0_torque, joint_1_torque])
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
    #     mu, cov = np.mean(base[:, 11:17], axis=0), np.cov(base[:, 11:17], rowvar=False)
    #     cen_dat = data[:, 11:17] - mu
    #     ll = -(cen_dat @ np.linalg.inv(cov) * cen_dat).mean(axis=-1) / 2
    #     return ll < self.thresh

    def is_different(self, data: np.ndarray, base: np.ndarray) -> np.ndarray:
        assert base.shape[-1] == 17
        if self.dist_func_type == 'l2':
            base_context = self.only_context(base)
            data_context = self.only_context(data)
            mu_base = np.r_[np.mean(base_context[:, :3]), np.mean(base_context[:, 3:])]
            mu_data = data_context.reshape((-1, 2, 3)).mean(axis=-1)
            dist = ((mu_data - mu_base) ** 2).sum(axis=-1)
            return dist > self.thresh
        elif self.dist_func_type == 'mahala':
            base_context = self.only_context(base)
            data_context = self.only_context(data)
            mu, cov = np.mean(base_context, axis=0), np.cov(base_context, rowvar=False)
            ind = cov.sum(axis=-1) > 0
            cov = cov[ind][:, ind]
            cen_dat = data_context[:, ind] - mu[ind]
            lu = np.linalg.cholesky(cov)
            y = solve_triangular(lu, cen_dat.T, lower=True)
            d = np.einsum('ij,ij->j', y, y)
            ll = -d / base.shape[-1] / 2
            return ll < self.thresh
        elif self.dist_func_type == 'mahala_full':
            mu, cov = np.mean(base, axis=0), np.cov(base, rowvar=False)
            ind = cov.sum(axis=-1) > 0
            cov = cov[ind][:, ind]
            cen_dat = data[:, ind] - mu[ind]
            lu = np.linalg.cholesky(cov)
            y = solve_triangular(lu, cen_dat.T, lower=True)
            d = np.einsum('ij,ij->j', y, y)
            ll = -d / base.shape[-1] / 2
            return ll < self.thresh
        else:
            raise ValueError(f'No such distance function type {self.dist_func_type}')

    @staticmethod
    def no_context_obs(obs: np.ndarray) -> np.ndarray:
        return obs[..., :11]

    @staticmethod
    def only_context(obs: np.ndarray) -> np.ndarray:
        return obs[..., 11:]

    @property
    def context_size(self):
        return 6
