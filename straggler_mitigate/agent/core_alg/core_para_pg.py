from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import value_train, policy_gradient
from param import config
from neural_net.nn_perm import PermInvNet


def cumulative_rewards(rewards_np: np.ndarray, dones_np: np.ndarray, next_value_np: np.ndarray, gamma_rate: float,
                       cuts_np: np.ndarray, times_np: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(rewards_np), dtype=np.float32)
    last_val = 0
    # if done is true (and 1), next val should be zero in return
    gamma_arr = np.exp(times_np * gamma_rate) * (1-dones_np.astype(float))

    for i in reversed(range(len(rewards_np))):
        if cuts_np[i]:
            last_val = next_value_np[i]
        last_val = rewards_np[i] + gamma_arr[i] * last_val
        returns[i] = last_val

    return returns


def gae_advantage(rewards_np: np.ndarray, dones_np: np.ndarray, values_np: np.ndarray, next_values_np: np.ndarray,
                  gamma_rate: float, cuts_np: np.ndarray, times_np: np.ndarray) -> np.ndarray:
    # TD lambda style advantage computation
    # more details in GAE: https://arxiv.org/pdf/1506.02438.pdf
    adv = np.zeros([len(rewards_np), 1], dtype=np.float32)
    last_gae = 0
    gamma_arr = np.exp(times_np * gamma_rate) * (1-dones_np.astype(float))

    for i in reversed(range(len(rewards_np))):
        if cuts_np[i]:
            last_gae = 0
        delta = rewards_np[i] + gamma_arr[i] * next_values_np[i] - values_np[i]
        last_gae = adv[i] = delta + gamma_arr[i] * config.lam * last_gae

    return adv


def train_actor_critic(value_net: PermInvNet or torch.nn.Module, policy_net: PermInvNet or torch.nn.Module,
                       net_opt_p: torch.optim.Optimizer, net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss,
                       device: torch.device, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray,
                       obs_np: np.ndarray, dones_np: np.ndarray, masks_np: np.ndarray, gamma_rate: float,
                       entropy_factor: float, cuts_np: np.ndarray, times_np: np.ndarray = None,
                       monitor: SummaryWriter = None, it: int = None) -> Tuple[float, float, float, np.ndarray,
                                                                               np.ndarray, float, np.ndarray]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    masks_torch = torch.as_tensor(masks_np, dtype=torch.float, device=device)

    # compute values
    values_torch = value_net.forward(obs_torch)
    values_np = values_torch.cpu().detach().numpy()
    next_values_torch = value_net.forward(torch.as_tensor(next_obs_np, dtype=torch.float, device=device))
    next_values_np = next_values_torch.cpu().detach().numpy()

    # cumulative reward
    returns_np = cumulative_rewards(rewards_np, dones_np, next_values_np, gamma_rate, cuts_np, times_np)
    returns_torch = torch.as_tensor(returns_np, dtype=torch.float, device=device)

    # compute advantage
    adv_np = gae_advantage(rewards_np, dones_np, values_np, next_values_np, gamma_rate, cuts_np, times_np)
    adv_torch = torch.as_tensor(adv_np, dtype=torch.float, device=device)

    # policy gradient training
    pg_loss, entropy, log_pi_min = \
        policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, masks_torch, entropy_factor,
                        monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
