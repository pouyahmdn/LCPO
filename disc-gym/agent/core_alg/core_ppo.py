from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import cumulative_rewards, gae_advantage, value_train, get_kl
from agent.core_alg.core_utils import get_flat_params_from
from neural_net.nn import FullyConnectNN, FCNPolicy


def proximal_policy_optimization(
        policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer, states_torch: torch.Tensor,
        actions_torch: torch.Tensor, adv_torch: torch.Tensor, entropy_factor: float, ppo_kl: float, ppo_iters: int,
        ppo_clip: float, monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    pg_loss_s = []
    entropy_s = []

    with torch.no_grad():
        log_pi_act_before, entropy_before, log_pi_before, pi_before = policy_net.full_act(states_torch, actions_torch)
        log_pi_min = log_pi_act_before.min().detach().item()
    prev_params = get_flat_params_from(policy_net)

    permutation = torch.randperm(states_torch.size()[0])
    permutation = permutation.repeat(3)
    batch_size = len(permutation) // ppo_iters
    permutation = permutation[:ppo_iters * batch_size].view(ppo_iters, batch_size)

    for i in range(ppo_iters):
        idx_s = permutation[i]

        log_pi_act, entropy, log_pi, pi = policy_net.full_act(states_torch[idx_s], actions_torch[idx_s])
        entropy_mean = entropy.mean()
        ratio = torch.exp(log_pi_act - log_pi_act_before[idx_s])
        clipped_ratio = ratio.clamp(min=1 - ppo_clip, max=1 + ppo_clip)
        pg_loss = - (torch.min(clipped_ratio * adv_torch[idx_s], ratio * adv_torch[idx_s])).mean()
        loss = pg_loss - entropy_factor * entropy_mean

        approx_kl = get_kl(log_pi_before[idx_s], pi_before[idx_s], log_pi).mean().item()
        if approx_kl > 1.5 * ppo_kl:
            break

        net_opt_p.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        net_opt_p.step()

        pg_loss_s.append(pg_loss.item())
        entropy_s.append(entropy_mean.item())

    log_pi_new = policy_net.log_pi(states_torch)
    kl = get_kl(log_pi_before, pi_before, log_pi_new)
    monitor.add_scalar('lpg/kl_in_d', kl.mean().item(), it)

    new_params = get_flat_params_from(policy_net)
    lr_approx = torch.abs(new_params-prev_params).mean()
    monitor.add_scalar('lpg/lr_approx', lr_approx, it)

    return np.mean(pg_loss_s).item(), np.mean(entropy_s).item(), log_pi_min


def train_ppo(value_net: FullyConnectNN, policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer,
              net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
              actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              terms_np: np.ndarray, truncs_np: np.ndarray, gamma: float, lam: float, entropy_factor: float,
              ppo_kl: float, ppo_iters: int, ppo_clip: float, monitor: SummaryWriter,
              it: int) -> Tuple[float, float, float, np.ndarray, np.ndarray,
                                                                         float, np.ndarray]:

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

    # policy gradient training
    pg_loss, entropy, log_pi_min = \
        proximal_policy_optimization(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, entropy_factor, ppo_kl,
                                     ppo_iters, ppo_clip, monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
