from typing import Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_utils import get_flat_params_from, get_kl
from neural_net.nn import FullyConnectNN, FCNPolicy


def clear_policy_gradient(policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer, states_torch: torch.Tensor,
                          actions_torch: torch.Tensor, adv_torch: torch.Tensor, log_probs_torch: torch.Tensor, p_co: float,
                          entropy_factor: float, monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    log_pi_act, entropy, log_pi, pi = policy_net.full_act(states_torch, actions_torch)
    log_pi_min = log_pi_act.min().detach().item()
    pg_loss = - (log_pi_act * adv_torch).mean()
    entropy_mean = entropy.mean()
    clone_loss = (torch.exp(torch.sum(log_probs_torch, dim=-1)) * log_pi_act).mean()
    loss = pg_loss - entropy_factor * entropy_mean + p_co * clone_loss

    prev_params = get_flat_params_from(policy_net)

    net_opt_p.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5, error_if_nonfinite=True)
    net_opt_p.step()

    log_pi_new = policy_net.log_pi(states_torch)
    kl = get_kl(log_pi, pi, log_pi_new)
    if monitor:
        monitor.add_scalar('lpg/kl_in_d', kl.mean().item(), it)

    new_params = get_flat_params_from(policy_net)
    lr_approx = torch.abs(new_params-prev_params).mean()
    if monitor:
        monitor.add_scalar('lpg/lr_approx', lr_approx, it)

    return pg_loss.item(), entropy_mean.item(), log_pi_min


def clear_value_train(value_net: FullyConnectNN or torch.nn.Module, net_opt_v: torch.optim.Optimizer,
                      net_loss: torch.nn.MSELoss, values_torch: torch.Tensor, v_s_torch: torch.Tensor,
                      past_values: torch.Tensor, v_co: float) -> float:
    n = len(values_torch) // 2
    v_loss = net_loss(values_torch, v_s_torch)
    v_loss += v_co * net_loss(values_torch[n:], past_values[n:])
    net_opt_v.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5, error_if_nonfinite=True)
    net_opt_v.step()
    return v_loss.item()


def v_trace(action_log_probs: np.ndarray, c: float, gamma: float, log_pi_act: np.ndarray, next_values_np: np.ndarray,
            rewards_np: np.ndarray, rho: float, truncs_np: np.ndarray, values_np: np.ndarray) -> Tuple[np.ndarray,
                                                                                                       np.ndarray,
                                                                                                       np.ndarray]:
    rho_vec = np.exp(np.minimum(np.log(rho), log_pi_act - action_log_probs.sum(axis=-1)))
    c_vec = np.exp(np.minimum(np.log(c), log_pi_act - action_log_probs.sum(axis=-1)))
    delta_v = rho_vec * (rewards_np + gamma * next_values_np - values_np)
    v_s = np.zeros(len(rewards_np), dtype=np.float32)
    next_v_s = np.zeros(len(rewards_np), dtype=np.float32)
    additive = 0
    for i in reversed(range(len(rewards_np))):
        if truncs_np[i] is True:
            additive = 0
        next_v_s[i] = next_values_np[i] + additive
        additive = additive * gamma * c_vec[i] + delta_v[i]
        v_s[i] = values_np[i] + additive
    return next_v_s, rho_vec, v_s


def train_clear(value_net: FullyConnectNN, policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer,
                net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
                actions_np: np.ndarray, action_log_probs: np.ndarray, action_values: np.ndarray, next_obs_np: np.ndarray,
                rewards_np: np.ndarray, obs_np: np.ndarray, terms_np: np.ndarray, truncs_np: np.ndarray, gamma: float,
                entropy_factor: float, c: float, rho: float, p_co: float, v_co: float,
                monitor: SummaryWriter, it: int) -> Tuple[float, float, float, np.ndarray, np.ndarray, float,
                                                    np.ndarray]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)

    # compute values
    values_torch = value_net.forward(obs_torch).squeeze()
    values_np = values_torch.cpu().detach().numpy()
    next_values_torch = value_net.forward(torch.as_tensor(next_obs_np, dtype=torch.float, device=device)).squeeze()
    next_values_np = next_values_torch.cpu().detach().numpy()

    log_pi_act, _, _, _ = policy_net.full_act(obs_torch, actions_torch)
    log_pi_act = log_pi_act.detach().numpy()
    next_v_s, rho_vec, v_s = v_trace(action_log_probs, c, gamma, log_pi_act, next_values_np, rewards_np, rho, truncs_np,
                                     values_np)

    # value training
    v_s_torch = torch.as_tensor(v_s, dtype=torch.float, device=device)
    past_values_torch = torch.as_tensor(action_values, dtype=torch.float, device=device)
    v_loss = clear_value_train(value_net, net_opt_v, net_loss, values_torch, v_s_torch, past_values_torch, v_co)

    adv_np = rewards_np + gamma * next_v_s - values_np
    adv_torch = torch.as_tensor(adv_np * rho_vec, dtype=torch.float, device=device)
    log_probs_torch = torch.as_tensor(action_log_probs, dtype=torch.float, device=device)

    # policy gradient training
    pg_loss, entropy, log_pi_min = \
        clear_policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, log_probs_torch, p_co,
                              entropy_factor, monitor, it)

    return pg_loss, v_loss, entropy, v_s, values_np, log_pi_min, adv_np


