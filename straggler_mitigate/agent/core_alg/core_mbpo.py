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
