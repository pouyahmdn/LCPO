from typing import Union, Tuple
import numpy as np
import torch

from param import config
from neural_net.nn_perm import PermInvNet


def sample_action(q_net: Union[PermInvNet, torch.nn.Module], obs: np.ndarray, device: torch.device) -> int:
    max_choice = q_net.max(torch.as_tensor(obs, dtype=torch.float, device=device))[1]
    return max_choice.item()


def soft_copy(critic_target: PermInvNet, critic_local: PermInvNet, tau: float):
    assert 0 <= tau <= 1
    for target_param, source_param in zip(critic_target.parameters(), critic_local.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def train_dqn(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              dones_np: np.ndarray, q_net: PermInvNet, target_net: PermInvNet, net_opt_q: torch.optim.Optimizer,
              net_loss: torch.nn.MSELoss or torch.nn.HuberLoss, device: torch.device, gamma_rate: float,
              mask_next_np: np.ndarray, mask_choice: int, times_np: np.ndarray = None) -> Tuple[float, np.ndarray,
                                                                                                np.ndarray]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    mask_next_torch = torch.as_tensor(mask_next_np, dtype=torch.bool, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    times_torch = torch.as_tensor(times_np, dtype=torch.float, device=device)
    dones_torch = torch.as_tensor(dones_np, dtype=torch.bool, device=device)

    with torch.no_grad():
        best_next_choices_q = q_net.max(next_obs_torch)[1]
        assert torch.equal(mask_next_torch, next_obs_torch[:, -1].bool())
        best_next_choices_q[mask_next_torch] = mask_choice

        val_target_best_next_choice = target_net.forward(next_obs_torch).gather(1, best_next_choices_q.unsqueeze(-1))
        val_target_best_next_choice = val_target_best_next_choice.squeeze()

        if times_torch is None:
            best_val_next_state = rewards_torch + (~dones_torch).float() * gamma_rate * val_target_best_next_choice
        else:
            best_val_next_state = rewards_torch + (~dones_torch).float() * torch.exp(times_torch * gamma_rate) * \
                                  val_target_best_next_choice

    val_q = q_net.forward(obs_torch).gather(1, actions_torch).squeeze()

    v_loss = net_loss(val_q, best_val_next_state)
    net_opt_q.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1)
    net_opt_q.step()

    soft_copy(target_net, q_net, config.off_policy_tau)

    return v_loss.item(), val_q.detach().cpu().numpy(), best_val_next_state.cpu().numpy()
