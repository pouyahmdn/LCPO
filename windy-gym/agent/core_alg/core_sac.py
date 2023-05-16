from typing import Tuple
import numpy as np
import torch

from agent.core_alg.core_dqn import soft_copy
from neural_net.nn import FCNPolicy


def policy_gradient(policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer, critic_1_local: FCNPolicy,
                    critic_2_local: FCNPolicy, states_torch: torch.Tensor, entropy_factor: float) -> Tuple[float,
                                                                                                           float]:
    # Entropy: Batch
    # log-pi: Batch X ACT-DIM X BINS
    # pi: Batch X ACT-DIM X BINS
    entropy, log_pi, pi = policy_net.dist(states_torch)

    with torch.no_grad():
        # Batch X ACT-DIM X BINS
        q_local_1 = critic_1_local(states_torch)
        # Batch X ACT-DIM X BINS
        q_local_2 = critic_2_local(states_torch)
        # Batch X ACT-DIM X BINS
        min_q_local = torch.min(q_local_1, q_local_2)
    # Sum(i.e. assuming action dimension effects on Q-value can be disaggregated): 1
    loss = ((log_pi * entropy_factor - min_q_local) * pi).sum(dim=(-1, -2)).mean(dim=0)

    net_opt_p.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    net_opt_p.step()

    return loss.item(), entropy.mean().item()


def value_train(policy_net: FCNPolicy, critic_1_target: FCNPolicy, critic_2_target: FCNPolicy,
                critic_1_local: FCNPolicy, critic_2_local: FCNPolicy, net_opt_v_1: torch.optim.Optimizer,
                net_opt_v_2: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, states_torch: torch.Tensor,
                next_states_torch: torch.Tensor, rewards_torch: torch.Tensor, terms_torch: torch.Tensor,
                actions_torch: torch.Tensor, gamma: float, entropy_factor: float) -> Tuple[float, float, np.ndarray,
                                                                                           np.ndarray, np.ndarray]:
    with torch.no_grad():
        # log-pi: Batch X ACT-DIM X BINS
        log_pi = policy_net.log_pi(states_torch)
        # pi: Batch X ACT-DIM X BINS
        pi = torch.exp(log_pi)
        # Batch X ACT-DIM X BINS
        next_q_target1 = critic_1_target(next_states_torch)
        # Batch X ACT-DIM X BINS
        next_q_target2 = critic_2_target(next_states_torch)
        # Batch X ACT-DIM X BINS
        min_qs_local = torch.min(next_q_target1, next_q_target2)
        # Batch X ACT-DIM X BINS
        min_next_q_target = pi * (min_qs_local - entropy_factor * log_pi)

        # Sum(i.e. assuming action dimension effects on Q-value can be disaggregated): Batch
        min_next_q_target = min_next_q_target.sum(dim=(-1, -2))
        # Batch
        q_target = rewards_torch + (~terms_torch).float() * gamma * min_next_q_target

    # Before Gather: Batch X ACT-DIM X BINS
    # After Gather: Batch X ACT-DIM X 1
    # Sum(i.e. assuming action dimension effects on Q-value can be disaggregated): Batch
    q_local_1 = critic_1_local(states_torch).gather(-1, actions_torch.unsqueeze(dim=-1)).squeeze(dim=-1).sum(dim=-1)
    # Before Gather: Batch X ACT-DIM X BINS
    # After Gather: Batch X ACT-DIM X 1
    # Sum(i.e. assuming action dimension effects on Q-value can be disaggregated): Batch
    q_local_2 = critic_2_local(states_torch).gather(-1, actions_torch.unsqueeze(dim=-1)).squeeze(dim=-1).sum(dim=-1)

    v_loss_1 = net_loss(q_local_1, q_target)
    net_opt_v_1.zero_grad()
    v_loss_1.backward()
    torch.nn.utils.clip_grad_norm_(critic_1_local.parameters(), 1)
    net_opt_v_1.step()

    v_loss_2 = net_loss(q_local_2, q_target)
    net_opt_v_2.zero_grad()
    v_loss_2.backward()
    torch.nn.utils.clip_grad_norm_(critic_2_local.parameters(), 1)
    net_opt_v_2.step()

    return v_loss_1.item(), v_loss_2.item(), q_target.detach().numpy(), q_local_1.detach().numpy(), \
        q_local_2.detach().numpy()


def train_sac(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
              terms_np: np.ndarray, truncs_np: np.ndarray, policy_net: FCNPolicy, critic_1_target: FCNPolicy,
              critic_2_target: FCNPolicy, critic_1_local: FCNPolicy, critic_2_local: FCNPolicy,
              net_opt_v_1: torch.optim.Optimizer, net_opt_v_2: torch.optim.Optimizer, net_opt_p: torch.optim.Optimizer,
              net_loss: torch.nn.MSELoss, device: torch.device, gamma: float, entropy_factor: float,
              tau: float) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    terms_torch = torch.as_tensor(terms_np, dtype=torch.bool, device=device)

    # policy gradient training
    pg_loss, entropy = policy_gradient(policy_net, net_opt_p, critic_1_local, critic_2_local, obs_torch, entropy_factor)

    # value training
    v_loss_1, v_loss_2, q_target, q_local_1, q_local_2 = \
        value_train(policy_net, critic_1_target, critic_2_target, critic_1_local, critic_2_local, net_opt_v_1,
                    net_opt_v_2, net_loss, obs_torch, next_obs_torch, rewards_torch, terms_torch, actions_torch, gamma,
                    entropy_factor)

    soft_copy(critic_1_target, critic_1_local, tau)
    soft_copy(critic_2_target, critic_2_local, tau)

    return pg_loss, v_loss_1, v_loss_2, entropy, q_target, q_local_1, q_local_2
