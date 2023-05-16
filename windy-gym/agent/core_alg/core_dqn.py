from typing import Tuple
import numpy as np
import torch

from neural_net.nn import FullyConnectNN, FCNPolicy


def soft_copy(critic_target: FullyConnectNN, critic_local: FullyConnectNN, tau: float):
    assert 0 <= tau <= 1
    for target_param, source_param in zip(critic_target.parameters(), critic_local.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def train_dqn(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              terms_np: np.ndarray, truncs_np: np.ndarray, q_net: FCNPolicy, target_net: FCNPolicy,
              net_opt_q: torch.optim.Optimizer, net_loss: torch.nn.MSELoss or torch.nn.HuberLoss, device: torch.device,
              gamma: float, tau: float) -> Tuple[float, np.ndarray,np.ndarray]:
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

    v_loss = net_loss(val_q, best_val_next_state)
    net_opt_q.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0, error_if_nonfinite=True)
    net_opt_q.step()

    soft_copy(target_net, q_net, tau)

    return v_loss.item(), val_q.detach().cpu().numpy(), best_val_next_state.cpu().numpy()
