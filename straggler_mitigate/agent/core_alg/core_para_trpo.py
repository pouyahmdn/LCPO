from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import value_train
from agent.core_alg.core_para_pg import cumulative_rewards, gae_advantage
from agent.core_alg.core_trpo import original_trpo
from neural_net.nn_perm import PermInvNet


def train_trpo(value_net: PermInvNet or torch.nn.Module, policy_net: PermInvNet or torch.nn.Module,
               net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
               actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
               dones_np: np.ndarray, masks_np: np.ndarray, gamma_rate: float, entropy_factor: float, cuts_np: np.ndarray,
               monitor: SummaryWriter, it: int, times_np: np.ndarray = None) -> Tuple[float, float, float, np.ndarray,
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

    pg_loss, entropy, log_pi_min = original_trpo(policy_net, obs_torch, actions_torch, adv_torch, masks_torch,
                                                 entropy_factor, monitor, it)

    # value training
    v_loss = -1
    for _ in range(20):
        v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)
        values_torch = value_net.forward(obs_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
