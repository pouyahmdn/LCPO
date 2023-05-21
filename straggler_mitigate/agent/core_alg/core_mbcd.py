import numpy as np
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch

from param import config

# Adapted from: https://github.com/LucasAlegre/mbcd


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


def construct_model(obs_dim, act_dim, rew_dim, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = [32, 32]
    dev = torch.device(config.device)
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
        self.len_act_space = len(config.lb_timeout_levels)
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

    def predict(self, in_vec: np.ndarray or torch.tensor, factored=True, to_numpy=True, normalize=True,
                ret_var=True) -> Tuple[torch.Tensor, torch.Tensor] or Tuple[np.ndarray, np.ndarray]:
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
                            F.one_hot(act, num_classes=self.len_act_space).squeeze()], dim=-1)

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


class MBCD:
    def __init__(self, state_dim, action_dim, memory_capacity=100000, cusum_threshold=300, max_std=0.5, num_stds=2):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory_capacity = memory_capacity
        self.memory = Dataset(state_dim, action_dim, memory_capacity)

        self.threshold = cusum_threshold
        self.max_std = max_std
        self.num_stds = num_stds
        self.min_steps = 5000
        self.changed = False
        self.step = 0

        self.num_models = 1
        self.current_model = 0
        self.steps_per_context = {0: 0}
        self.models = {0: self._build_model()}
        self.log_prob = {0: 0.0}
        self.var_mean = {0: 0.0}
        self.new_model_log_pdf = 0
        self.S = {0: 0.0, -1: 0.0}  # -1 is statistic for new model
        self.mean, self.variance = {}, {}

        self.test_mode = False

    @property
    def counter(self):
        return self.steps_per_context[self.current_model]

    def _build_model(self):
        return construct_model(obs_dim=self.state_dim, act_dim=self.action_dim, rew_dim=1, hidden_dim=[64, 64, 64])

    def train(self):
        x, y = self.memory.to_train_batch(5 * 256)
        print('Training model', self.current_model)
        self.models[self.current_model].train_model(x, y, batch_size=256)

    def get_logprob2(self, x, means, variances):
        k = x.shape[-1]

        mean = np.mean(means, axis=0)
        variance = (np.mean(means ** 2 + variances, axis=0) - mean ** 2) + 1e-6

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (
                k * np.log(2 * np.pi) + np.log(variance).sum(-1) + (np.power(x - mean, 2) / variance).sum(-1))
        assert len(log_prob) == 1
        log_prob = log_prob.sum(0)

        # numerical instability filter is limiting log_prob to np.log(1e-8), plus batch_size is always 1

        # ## [ batch_size ]
        # prob = np.exp(log_prob).sum(axis=0)
        #
        # ## [ batch_size ]
        # log_prob = np.log(prob + 1e-8)  # Avoid log of zero

        var_mean = np.linalg.norm(np.std(means, axis=0), axis=-1)

        return log_prob, var_mean[0], mean, variance

    def update_metrics(self, obs, action, reward, next_obs, done):
        obs = obs[None]
        action = action[None]
        inputs = np.concatenate((obs, action), axis=-1)
        true_output = np.concatenate(([reward], next_obs))[None]

        preds = []
        if self.changed:
            self.S = {m: 0.0 for m in range(self.num_models)}
            self.S[-1] = 0.0

        for i in range(self.num_models):
            model_means, model_vars = self.models[i].predict(inputs, factored=True)
            preds.append(model_means.copy())
            model_means[:, :, 1:] += obs
            self.log_prob[i], self.var_mean[i], self.mean[i], self.variance[i] = self.get_logprob2(true_output,
                                                                                                   model_means,
                                                                                                   model_vars)

        for i in (m for m in range(self.num_models) if m != self.current_model):
            if self.var_mean[self.current_model] < self.max_std and self.counter > self.min_steps:
                log_ratio = self.log_prob[i] - self.log_prob[self.current_model]  # log(a/b) = log(a) - log(b)
                self.S[i] = max(0, self.S[i] + log_ratio)

        # Save to object so we can log it in tensorboard
        new_model_log_pdf = -1 / 2 * ((self.state_dim + 1) * np.log(2 * np.pi) +
                                      np.log(self.variance[self.current_model]).sum(-1) +
                                      (np.power(true_output - (true_output + self.num_stds * np.sqrt(
                                          self.variance[self.current_model])), 2) / self.variance[
                                           self.current_model]).sum(-1))
        assert len(new_model_log_pdf) == 1
        new_model_log_pdf = new_model_log_pdf.sum(0)

        # Save to object so we can log it in tensorboard
        self.new_model_log_pdf = new_model_log_pdf
        # self.new_model_log_pdf = np.log(np.exp(self.new_model_log_pdf).sum(0) + 1e-8)

        if self.var_mean[self.current_model] < self.max_std and self.counter > self.min_steps:
            log_ratio = new_model_log_pdf - self.log_prob[self.current_model]
            self.S[-1] = max(0, self.S[-1] + log_ratio)

        changed = False
        maxm = max(self.S.values())
        if maxm > self.threshold:
            changed = True
            self.memory.remove_last_n(n=100)  # Remove last experiences, as they may be from different context

            if maxm == self.S[-1]:  # New Model
                newm = self.new_model()
                self.set_model(newm, load_params_from_init_model=True)
            else:
                newm = max(self.S, key=lambda key: self.S[key])
                self.set_model(newm)

        self.changed = changed
        self.step += 1
        self.steps_per_context[self.current_model] += 1

        return changed

    def predict(self, sa, model=None):
        model = self.current_model if model is None else model
        mean, var = self.models[model].predict(sa)
        return mean

    def new_model(self):
        self.steps_per_context[self.num_models] = 0
        self.models[self.num_models] = self._build_model()
        self.log_prob[self.num_models] = 0.0
        self.S[self.num_models] = 0.0
        self.var_mean[self.num_models] = 0.0
        self.num_models += 1
        return self.num_models - 1

    def set_model(self, model_id, load_params_from_init_model=False):
        self.current_model = model_id
        # new model
        if load_params_from_init_model:
            self.memory = Dataset(self.state_dim, self.action_dim, self.memory_capacity)
        # load existent model

    def add_experience(self, obs, actions, rewards, next_obs, dones):
        self.memory.push(obs, actions, rewards, next_obs, dones)
