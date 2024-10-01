from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_utils import get_flat_params_from, get_kl
from neural_net.nn import FullyConnectNN, FCNPolicy


def cumulative_rewards(rewards_np: np.ndarray, term_np: np.ndarray, trunc_np: np.ndarray,
                       next_value_np: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards_np), dtype=np.float32)
    last_val = next_value_np[-1]
    # if done is true (and 1), next val should be zero in return

    for i in reversed(range(len(rewards_np))):
        if trunc_np[i] is True:
            last_val = next_value_np[i]
        returns[i] = last_val = rewards_np[i] + gamma * last_val * (~term_np[i]).astype(float)

    return returns


def gae_advantage(rewards_np: np.ndarray, term_np: np.ndarray, trunc_np: np.ndarray, values_np: np.ndarray,
                  next_values_np: np.ndarray, gamma: float, lamb: float) -> np.ndarray:
    # TD lambda style advantage computation
    # more details in GAE: https://arxiv.org/pdf/1506.02438.pdf
    adv = np.zeros(len(rewards_np), dtype=np.float32)
    td0 = rewards_np + gamma * next_values_np * (~term_np).astype(float) - values_np
    last_gae = 0

    for i in reversed(range(len(rewards_np))):
        if trunc_np[i] is True:
            last_gae = 0
        last_gae = adv[i] = td0[i] + gamma * lamb * last_gae * (~term_np[i]).astype(float)

    return adv


def policy_gradient(policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer,
                    states_torch: torch.Tensor, actions_torch: torch.Tensor, adv_torch: torch.Tensor,
                    entropy_factor: float, monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    log_pi_act, entropy, log_pi, pi = policy_net.full_act(states_torch, actions_torch)
    log_pi_min = log_pi_act.min().detach().item()
    pg_loss = - (log_pi_act * adv_torch).mean()
    entropy_mean = entropy.mean()
    loss = pg_loss - entropy_factor * entropy_mean

    prev_params = get_flat_params_from(policy_net)

    net_opt_p.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5, error_if_nonfinite=True)
    net_opt_p.step()

    log_pi_new = policy_net.log_pi(states_torch)
    kl = get_kl(log_pi, pi, log_pi_new)
    monitor.add_scalar('lpg/kl_in_d', kl.mean().item(), it)

    new_params = get_flat_params_from(policy_net)
    lr_approx = torch.abs(new_params-prev_params).mean()
    monitor.add_scalar('lpg/lr_approx', lr_approx, it)

    return pg_loss.item(), entropy_mean.item(), log_pi_min


def get_kl(log_pi_old: torch.Tensor, pi_old: torch.Tensor, log_pi_new: torch.Tensor) -> torch.Tensor:
    kl = (pi_old * (log_pi_old - log_pi_new)).sum(dim=(-1, -2))
    return kl


def value_train(value_net: FullyConnectNN or torch.nn.Module, net_opt_v: torch.optim.Optimizer,
                net_loss: torch.nn.MSELoss, values_torch: torch.Tensor, returns_torch: torch.Tensor) -> float:
    v_loss = net_loss(values_torch, returns_torch)
    net_opt_v.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5, error_if_nonfinite=True)
    net_opt_v.step()
    return v_loss.item()


def train_actor_critic(value_net: FullyConnectNN, policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer,
                       net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
                       actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
                       terms_np: np.ndarray, truncs_np: np.ndarray, gamma: float, lam: float, entropy_factor: float,
                       monitor: SummaryWriter, it: int) -> Tuple[float, float, float, np.ndarray, np.ndarray, float,
                                                                 np.ndarray]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)

    # compute values
    values_torch = value_net.forward(obs_torch).squeeze()
    values_np = values_torch.cpu().detach().numpy()
    next_values_torch = value_net.forward(torch.as_tensor(next_obs_np, dtype=torch.float, device=device)).squeeze()
    next_values_np = next_values_torch.cpu().detach().numpy()

    # cumulative reward
    returns_np = cumulative_rewards(rewards_np, terms_np, truncs_np, next_values_np, gamma)
    returns_torch = torch.as_tensor(returns_np, dtype=torch.float, device=device)

    # compute advantage
    adv_np = gae_advantage(rewards_np, terms_np, truncs_np, values_np, next_values_np, gamma, lam)
    adv_torch = torch.as_tensor(adv_np, dtype=torch.float, device=device)
    # returns_np = adv_np + values_np
    # returns_torch = torch.as_tensor(ret_gae_np, dtype=torch.float, device=device)

    # policy gradient training
    pg_loss, entropy, log_pi_min = \
        policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, entropy_factor, monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np


def train_entropy(policy_net: FCNPolicy, obs_np: np.ndarray, log_entropy: torch.Tensor,
                  opt_ent: torch.optim.Optimizer, device: torch.device, target_entropy: float,
                  entropy_max: float) -> Tuple[float, float]:
    with torch.no_grad():
        obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
        entropy = policy_net.entropy(obs_torch)
        entropy_with_target = -entropy + target_entropy
    ent_loss = -(torch.exp(log_entropy) * entropy_max * entropy_with_target).mean()
    opt_ent.zero_grad()
    ent_loss.backward()
    opt_ent.step()
    entropy_factor = torch.exp(log_entropy).item() * entropy_max

    return ent_loss.item(), entropy_factor


