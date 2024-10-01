from typing import Tuple
import numpy as np
import torch

from neural_net.nn import FCNPolicy
from agent.core_alg.core_dqn import soft_copy
from agent.core_alg.core_utils import get_flat_params_from, set_flat_params_to


def train_bfdqn(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
                terms_np: np.ndarray, truncs_np: np.ndarray, q_net: FCNPolicy, target_net: FCNPolicy,
                net_opt_q: torch.optim.Optimizer, net_loss: torch.nn.MSELoss or torch.nn.HuberLoss,
                device: torch.device, gamma: float, tau: float, bf_n: int, bf_g: np.ndarray, bf_c: np.ndarray,
                bf_vars: np.ndarray, it: int) -> Tuple[float, np.ndarray, np.ndarray, torch.Tensor]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    terms_torch = torch.as_tensor(terms_np, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Batch X ACT-DIM X 1
        _, best_next_choices_q = q_net.max(next_obs_torch)

        # Before Gather: Batch X ACT-DIM X BINS
        # After Gather: Batch X ACT-DIM X 1
        val_target_best_next_choice = target_net.forward(next_obs_torch).gather(-1, best_next_choices_q.unsqueeze(-1))
        val_target_best_next_choice = val_target_best_next_choice.squeeze(-1).sum(dim=-1)
        # Sum(i.e. assuming action dimension effects on Q-value can be disaggregated): Batch

        # Batch
        best_val_next_state = rewards_torch + (~terms_torch).float() * gamma * val_target_best_next_choice

    # Before Gather: Batch X ACT-DIM X BINS
    # After Gather: Batch X ACT-DIM X 1
    # Sum(i.e. assuming action dimension effects on Q-value can be disaggregated): Batch
    val_q = q_net.forward(obs_torch).gather(-1, actions_torch.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    q_net_prior = get_flat_params_from(q_net)

    v_loss = net_loss(val_q, best_val_next_state)
    net_opt_q.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0, error_if_nonfinite=True)
    net_opt_q.step()

    w_ext = get_flat_params_from(q_net) - q_net_prior

    if bf_vars is None:
        n_params = w_ext.shape[0]
        bf_vars = torch.zeros((bf_n, n_params))
        bf_vars[0] = q_net_prior
        for i in range(1, bf_n):
            bf_vars[i] = torch.normal(0, 1/np.power(2, i).item(), size=(n_params, ))

    leak_vars = torch.zeros_like(bf_vars)

    leak_vars[0] = w_ext
    for i in range(1, bf_n):
        leak_vars[i] = bf_g[i-1] / bf_c[i] * (bf_vars[i-1]-bf_vars[i])

    for i in range(bf_n-1):
        if it > np.power(2, i) / bf_g[0]:
            leak_vars[i] += bf_g[i] / bf_c[i] * (bf_vars[i+1]-bf_vars[i])
    if it > np.power(2, bf_n-1) / bf_g[0]:
        leak_vars[bf_n-1] += bf_g[bf_n-1] / bf_c[bf_n-1] * -bf_vars[bf_n-1]

    bf_vars += leak_vars
    set_flat_params_to(q_net, bf_vars[0])

    soft_copy(target_net, q_net, tau)

    return v_loss.item(), val_q.detach().cpu().numpy(), best_val_next_state.cpu().numpy(), bf_vars
