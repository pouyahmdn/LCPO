from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as tfunctional
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_utils import get_flat_params_from
from param import config
from neural_net.nn_perm import PermInvNet


def sample_action(policy_net: PermInvNet or torch.nn.Module, obs: np.ndarray, device: torch.device) -> int:
    pi_cpu = policy_net.sample_policy(torch.as_tensor(obs, dtype=torch.float, device=device))
    act = (pi_cpu[:-1].cumsum(-1) <= torch.rand(1)).sum().item()
    return act


def cumulative_rewards(rewards_np: np.ndarray, dones_np: np.ndarray, last_next_value_np: float, gamma_rate: float,
                       times_np: np.ndarray = None) -> np.ndarray:
    returns = np.zeros(len(rewards_np), dtype=np.float32)
    last_val = last_next_value_np
    # if done is true (and 1), next val should be zero in return
    if times_np is None:
        gamma_arr = gamma_rate * (1-dones_np.astype(float))
    else:
        gamma_arr = np.exp(times_np * gamma_rate) * (1-dones_np.astype(float))

    for i in reversed(range(len(rewards_np))):
        last_val = rewards_np[i] + gamma_arr[i] * last_val
        returns[i] = last_val

    return returns


def gae_advantage(rewards_np: np.ndarray, dones_np: np.ndarray, values_np: np.ndarray, next_values_np: np.ndarray,
                  gamma_rate: float, times_np: np.ndarray = None) -> np.ndarray:
    # TD lambda style advantage computation
    # more details in GAE: https://arxiv.org/pdf/1506.02438.pdf
    adv = np.zeros([len(rewards_np), 1], dtype=np.float32)
    last_gae = 0
    if times_np is None:
        gamma_arr = gamma_rate * (1-dones_np.astype(float))
    else:
        gamma_arr = np.exp(times_np * gamma_rate) * (1-dones_np.astype(float))

    for i in reversed(range(len(rewards_np))):
        delta = rewards_np[i] + gamma_arr[i] * next_values_np[i] - values_np[i]
        last_gae = adv[i] = delta + gamma_arr[i] * config.lam * last_gae

    return adv


def policy_gradient(policy_net: PermInvNet or torch.nn.Module, net_opt_p: torch.optim.Optimizer,
                    states_torch: torch.Tensor, actions_torch: torch.Tensor, adv_torch: torch.Tensor,
                    mask: torch.Tensor, entropy_factor: float,
                    monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    q = policy_net.forward(states_torch)
    log_pi = tfunctional.log_softmax(q, dim=-1)
    log_pi_acts = log_pi.gather(1, actions_torch)

    pi = torch.exp(log_pi)

    mask = 1 - mask

    log_pi_min = (log_pi_acts * mask).min().detach().item()
    entropy = (log_pi * pi * mask).sum(dim=-1).mean()
    pg_loss = - (log_pi_acts * adv_torch * mask).mean()
    loss = pg_loss + entropy_factor * entropy

    prev_params = get_flat_params_from(policy_net)

    net_opt_p.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    net_opt_p.step()

    log_pi_new = tfunctional.log_softmax(policy_net.forward(states_torch), dim=-1)
    kl = (pi * (log_pi - log_pi_new)).sum(dim=-1)
    monitor.add_scalar('lpg/kl_in_d', kl.mean().item(), it)

    new_params = get_flat_params_from(policy_net)
    lr_approx = torch.abs(new_params-prev_params).mean()
    monitor.add_scalar('lpg/lr_approx', lr_approx, it)

    return pg_loss.item(), entropy.item(), log_pi_min


def value_train(value_net: PermInvNet or torch.nn.Module, net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss,
                values_torch: torch.Tensor, returns_torch: torch.Tensor) -> float:
    v_loss = net_loss(values_torch, returns_torch)
    net_opt_v.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1)
    net_opt_v.step()
    return v_loss.item()


def train_actor_critic(value_net: PermInvNet or torch.nn.Module, policy_net: PermInvNet or torch.nn.Module,
                       net_opt_p: torch.optim.Optimizer, net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss,
                       device: torch.device, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray,
                       obs_np: np.ndarray, dones_np: np.ndarray, masks_np: np.ndarray, gamma_rate: float,
                       entropy_factor: float, times_np: np.ndarray = None, monitor: SummaryWriter = None,
                       it: int = None) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, np.ndarray]:

    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    masks_torch = torch.as_tensor(masks_np, dtype=torch.float, device=device)

    # compute values
    values_torch = value_net.forward(obs_torch)
    values_np = values_torch.cpu().detach().numpy()
    next_values_torch = value_net.forward(torch.as_tensor(next_obs_np, dtype=torch.float, device=device))
    next_values_np = next_values_torch.cpu().detach().numpy()

    # cumulative reward
    returns_np = cumulative_rewards(rewards_np, dones_np, next_values_np[-1], gamma_rate, times_np)
    returns_torch = torch.as_tensor(returns_np, dtype=torch.float, device=device)

    # compute advantage
    adv_np = gae_advantage(rewards_np, dones_np, values_np, next_values_np, gamma_rate, times_np)
    adv_torch = torch.as_tensor(adv_np, dtype=torch.float, device=device)

    # policy gradient training
    pg_loss, entropy, log_pi_min = \
        policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, masks_torch, entropy_factor,
                        monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np


def train_entropy(policy_net: PermInvNet, obs_np: np.ndarray, log_entropy: torch.Tensor, opt_ent: torch.optim.Optimizer,
                  device: torch.device, target_entropy: float) -> Tuple[float, float]:
    with torch.no_grad():
        obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
        q = policy_net.forward(obs_torch)
        log_pi = tfunctional.log_softmax(q, dim=-1)
        pi = torch.exp(log_pi)
        entropy_with_target = (pi * log_pi).sum(dim=-1) + target_entropy
    ent_loss = -(torch.exp(log_entropy) * config.entropy_max * entropy_with_target).mean()
    opt_ent.zero_grad()
    ent_loss.backward()
    opt_ent.step()
    entropy_factor = torch.exp(log_entropy).item() * config.entropy_max

    return ent_loss.item(), entropy_factor
