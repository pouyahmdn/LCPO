from typing import Tuple
import numpy as np

from agent.core_alg.core_mbcd import Dataset, construct_model


class MBPO:
    def __init__(self, state_dim: int, action_dim: int, memory_capacity: int = 100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.memory = Dataset(state_dim, action_dim, memory_capacity)
        self.steps = 0
        self.model = construct_model(obs_dim=self.state_dim, act_dim=self.action_dim, rew_dim=1, hidden_dim=[64, 64, 64])
        self.log_prob_list = []
        self.var_list = []

    @property
    def counter(self) -> int:
        return self.steps

    def train(self):
        x, y = self.memory.to_train_batch(5*256)
        self.model.train_model(x, y, batch_size=256)

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        next_obs, rew = self.model.predict_next(obs, action)
        return next_obs.detach().cpu().numpy(), rew.detach().cpu().numpy()

    def add_experience(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray,
                       dones: np.ndarray):
        self.memory.push(obs, actions, rewards, next_obs, dones)

        assert obs.ndim == 1
        assert actions.ndim == actions.ndim
        inputs = np.concatenate((obs, actions), axis=-1)
        true_output = np.concatenate(([rewards], next_obs))

        model_means, model_vars = self.model.predict(inputs, factored=True)
        assert model_means.shape[1] == 1
        assert model_vars.shape[1] == 1
        model_means[:, :, 1:] += obs
        log_prob = -0.5 * (np.log(2 * np.pi) + np.log(model_vars) + (np.power(true_output - model_means, 2) / model_vars))
        self.log_prob_list.append(log_prob[:, 0, :])
        self.var_list.append(model_vars[:, 0, :])

    def get_accuracy(self) -> Tuple[np.array, np.array]:
        rets = (np.array(self.log_prob_list), np.array(self.var_list))
        self.log_prob_list.clear()
        self.var_list.clear()
        return rets


def angle_normalize(x: np.ndarray or float) -> np.ndarray or float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class OracleMBPO:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = 0
        self.var_list = np.ones((200, 5, self.state_dim+1)) * 1e-8
        self.log_prob_list = -0.5 * (np.log(2 * np.pi) + np.log(self.var_list))

        self.max_speed = 8
        self.max_torque = 2.0
        self.act_bins = np.linspace(-self.max_torque, self.max_torque, action_dim, endpoint=True)
        self.rng = np.random.RandomState(1234)

    @property
    def counter(self) -> int:
        return self.steps

    def train(self):
        pass

    def predict(self, obs: np.ndarray, action: np.ndarray, real_next_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        th = np.arctan2(obs[:, 1], obs[:, 0])
        thdot = obs[:, 2]

        act_cnt = self.act_bins[action][:, 0]
        curr_wind = [real_next_obs[:, 3], real_next_obs[:, 6]]

        act_n_wind = act_cnt + curr_wind[0] * obs[:, 0] + curr_wind[1] * obs[:, 1]
        act_n_wind = np.clip(act_n_wind, -self.max_torque, self.max_torque)
        act_cnt = np.clip(act_cnt, -self.max_torque, self.max_torque)

        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (act_cnt ** 2)

        newthdot = thdot + (15 * obs[:, 1] + 3.0 * act_n_wind) * 0.05
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * 0.05

        new_obs = np.c_[
            np.cos(newth),
            np.sin(newth),
            newthdot,
            real_next_obs[:, 3],
            obs[:, 3],
            obs[:, 4],
            real_next_obs[:, 6],
            obs[:, 6],
            obs[:, 7],
        ]

        return new_obs, -costs

    def add_experience(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray,
                       dones: np.ndarray):
        pass

    def get_accuracy(self) -> Tuple[np.array, np.array]:
        return self.log_prob_list, self.var_list
