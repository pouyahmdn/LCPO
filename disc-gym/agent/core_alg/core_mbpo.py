from typing import Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

# Adapted from: https://github.com/LucasAlegre/mbcd

def construct_model(obs_dim, act_dim, rew_dim, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = [32, 32]
    dev = torch.device('cpu')
    torch.set_num_threads(4)
    model = FCMBCD(obs_dim + act_dim, hidden_dim, obs_dim + rew_dim, ensemble_size=5, act=nn.ReLU, device=dev).to(dev)
    return model


class FCMBCD(nn.Module):
    def __init__(self, in_size, n_hids, out_size, device, ensemble_size, act=nn.ReLU):
        super(FCMBCD, self).__init__()

        self.in_size = in_size
        self.n_hids = n_hids
        self.ensemble_size = ensemble_size
        self.act = act
        self.out_size = out_size
        self.net_loss = torch.nn.MSELoss(reduction='mean')
        self.device = device

        self.mu = torch.zeros(in_size, device=self.device)
        self.sigma = torch.ones(in_size, device=self.device)

        self.max_logvar = (
                    torch.ones((1, 1, self.out_size), dtype=torch.float, device=self.device) * 0.5).requires_grad_()
        self.min_logvar = (
                    -torch.ones((1, 1, self.out_size), dtype=torch.float, device=self.device) * 10).requires_grad_()
        self.sp = torch.nn.Softplus()

        # parameter dimensions
        layers = [self.in_size]
        layers.extend(self.n_hids)
        layers.append(self.out_size * 2)

        # initialize layer operations
        self.modules = []
        for net in range(self.ensemble_size):
            self.modules.append([])
            for layer_idx in range(len(layers) - 1):
                self.modules[-1].append(nn.Linear(layers[layer_idx], layers[layer_idx + 1]))
                if layer_idx < len(layers) - 2:
                    self.modules[-1].append(self.act())

        # for forward function
        self.modules = [nn.Sequential(*self.modules[i]) for i in range(self.ensemble_size)]
        for mod in self.modules:
            mod.to(self.device)
        params_opt = sum([list(self.modules[i].parameters()) for i in range(self.ensemble_size)], [])
        params_opt += [self.max_logvar, self.min_logvar]

        self.opt = torch.optim.Adam(params_opt, lr=1e-3, weight_decay=1e-4)

    def forward(self, in_vec):
        in_vec = (in_vec - self.mu) / self.sigma
        out_vec = torch.cat([self.modules[i](in_vec).view(1, -1, 2 * self.out_size) for i in range(self.ensemble_size)],
                            dim=0)
        return out_vec

    def predict(self, in_vec: np.ndarray or torch.tensor, factored=True, to_numpy=True, normalize=True, ret_var=True) -> \
    Tuple[torch.Tensor, torch.Tensor] or Tuple[np.ndarray, np.ndarray]:
        assert factored
        if isinstance(in_vec, np.ndarray):
            in_vec = torch.as_tensor(in_vec, dtype=torch.float, device=self.device)
        if normalize:
            in_vec = (in_vec - self.mu) / self.sigma
        out_vec = torch.cat([self.modules[i](in_vec).view(1, -1, 2 * self.out_size) for i in range(self.ensemble_size)],
                            dim=0)

        logvar = self.max_logvar - self.sp(self.max_logvar - out_vec[:, :, self.out_size:])
        logvar = self.min_logvar + self.sp(logvar - self.min_logvar)

        model_means = out_vec[:, :, :self.out_size]
        model_vars = logvar

        if ret_var:
            model_vars = torch.exp(model_vars)

        if to_numpy:
            model_means = model_means.detach().cpu().numpy()
            model_vars = model_vars.detach().cpu().numpy()

        return model_means, model_vars

    def predict_next(self, obs: np.ndarray or torch.tensor, act: np.ndarray or torch.tensor) -> Tuple[torch.Tensor,
                                                                                                      torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        if isinstance(act, np.ndarray):
            act = torch.as_tensor(act, dtype=torch.int64, device=self.device)

        assert obs.ndim == act.ndim
        assert obs.ndim == 1 or obs.shape[0] == act.shape[0]
        in_vec = torch.cat([obs.view((-1, obs.shape[-1])),
                            F.one_hot(act, num_classes=15).view((-1, act.shape[-1] * 15))], dim=-1)

        in_vec = (in_vec - self.mu) / self.sigma
        ind = torch.randint(self.ensemble_size, size=(1,))
        out_vec = self.modules[ind](in_vec)
        assert out_vec.shape == (in_vec.shape[0], 2 * self.out_size)

        logvar = self.max_logvar - self.sp(self.max_logvar - out_vec[:, self.out_size:])
        logvar = self.min_logvar + self.sp(logvar - self.min_logvar)

        model_mean = out_vec[:, :self.out_size]
        model_std = torch.exp(logvar / 2)

        sample = model_mean + model_std * torch.normal(mean=0.0, std=1.0, size=(in_vec.shape[0], self.out_size))
        return sample[0, :, 1:] + obs, sample[0, :, 0]

    def single_train(self, x: np.ndarray, y: np.ndarray):
        y_pred, log_var_pred = self.predict(x, to_numpy=False, normalize=False, ret_var=False)
        inv_var_pred = torch.exp(-log_var_pred)

        mse_losses = (torch.square(y_pred - y) * inv_var_pred).mean()

        var_losses = log_var_pred.mean()
        total_losses = mse_losses + var_losses
        total_losses += 0.01 * self.max_logvar.mean() - 0.01 * self.min_logvar.mean()

        self.opt.zero_grad()
        total_losses.backward()
        self.opt.step()

    def train_model(self, x_in: np.ndarray, y_out: np.ndarray, batch_size=256, num_epochs=5):
        mu = np.mean(x_in, axis=0, keepdims=True)
        sigma = np.std(x_in, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        x_in = (x_in - mu) / sigma

        self.mu = torch.as_tensor(mu, dtype=torch.float, device=self.device)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float, device=self.device)

        permutation = np.random.permutation(x_in.shape[0])
        inputs = torch.as_tensor(x_in[permutation], dtype=torch.float, device=self.device)
        targets = torch.as_tensor(y_out[permutation], dtype=torch.float, device=self.device)

        for epoch in range(num_epochs):
            for batch_num in range(int(np.ceil(inputs.shape[0] / batch_size))):
                batch_idxs = slice(batch_num * batch_size, (batch_num + 1) * batch_size)
                self.single_train(inputs[batch_idxs], targets[batch_idxs])


class Dataset:

    def __init__(self, obs_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rews_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def remove_last_n(self, n):
        self.ptr -= n

    def sample(self, batch_size, replace=True):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        return self.obs_buf[inds], self.acts_buf[inds], self.rews_buf[inds], self.next_obs_buf[inds], self.done_buf[
            inds]

    def to_train_batch(self, batch_size, normalization=False):
        inds = np.random.choice(self.size, batch_size, replace=True)

        X = np.hstack((self.obs_buf[inds], self.acts_buf[inds]))
        Y = np.hstack((self.rews_buf[inds], self.next_obs_buf[inds] - self.obs_buf[inds]))

        return X, Y

    def __len__(self):
        return self.size


def normalize(data, mean, std):
    return (data - mean) / (std + 1e-10)


def denormalize(data, mean, std):
    return data * (std + 1e-10) + mean


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

