from typing import Tuple, List
import numpy as np
import torch

from .core_dqn import soft_copy
from neural_net.nn import FCNPolicy
from agent.core_alg.core_utils import get_flat_params_from, set_flat_params_to


def gram_schmidt(new_grad: torch.Tensor, past_grads: List[torch.Tensor],
                 ogd_n: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    with torch.no_grad():
        for v in past_grads:
            new_grad -= torch.dot(new_grad, v) / torch.dot(v, v) * v
        new_grad /= torch.sqrt(torch.dot(new_grad, new_grad))
        past_grads.append(new_grad)
        if len(past_grads) == ogd_n:
            past_grads = past_grads[1:]
    return new_grad, past_grads


def apply_ogd(loss: torch.Tensor, nn_module: FCNPolicy, past_grads: List[torch.Tensor], ogd_alpha: float,
              ogd_n: int) -> List[torch.Tensor]:
    new_grads = torch.autograd.grad(loss, nn_module.parameters(), create_graph=True)
    new_grads = torch.cat([grad.view(-1) for grad in new_grads])
    new_grads, past_grads = gram_schmidt(new_grads, past_grads, ogd_n)
    prev_params = get_flat_params_from(nn_module)
    set_flat_params_to(nn_module, prev_params + ogd_alpha * new_grads)
    return past_grads


def policy_gradient_ogd(policy_net: FCNPolicy, critic_1_local: FCNPolicy, critic_2_local: FCNPolicy,
                        states_torch: torch.Tensor, entropy_factor: float, past_grads: List[torch.Tensor],
                        ogd_alpha: float, ogd_n: int) -> Tuple[float, float, List[torch.Tensor]]:
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

    past_grads = apply_ogd(loss, policy_net, past_grads, ogd_alpha, ogd_n)

    return loss.item(), entropy.mean().item(), past_grads


def value_train_ogd(policy_net: FCNPolicy, critic_1_target: FCNPolicy, critic_2_target: FCNPolicy,
                    critic_1_local: FCNPolicy, critic_2_local: FCNPolicy, net_loss: torch.nn.MSELoss,
                    states_torch: torch.Tensor, next_states_torch: torch.Tensor, rewards_torch: torch.Tensor,
                    terms_torch: torch.Tensor, actions_torch: torch.Tensor, gamma: float, entropy_factor: float,
                    past_grads_1: List[torch.Tensor], past_grads_2: List[torch.Tensor], ogd_alpha: float, ogd_n: int) -> \
        Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, List[torch.Tensor], List[torch.Tensor]]:
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
    past_grads_1 = apply_ogd(v_loss_1, critic_1_local, past_grads_1, ogd_alpha, ogd_n)

    v_loss_2 = net_loss(q_local_2, q_target)
    past_grads_2 = apply_ogd(v_loss_2, critic_2_local, past_grads_2, ogd_alpha, ogd_n)

    return v_loss_1.item(), v_loss_2.item(), q_target.detach().numpy(), q_local_1.detach().numpy(), \
        q_local_2.detach().numpy(), past_grads_1, past_grads_2


def train_sliding_ogd(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
                      terms_np: np.ndarray, truncs_np: np.ndarray, policy_net: FCNPolicy, critic_1_target: FCNPolicy,
                      critic_2_target: FCNPolicy, critic_1_local: FCNPolicy, critic_2_local: FCNPolicy,
                      net_loss: torch.nn.MSELoss, device: torch.device, gamma: float, entropy_factor: float,
                      tau: float, past_grads: List[List[torch.Tensor]], ogd_alpha: float, ogd_n: int) -> Tuple[float,
float, float, float, np.ndarray, np.ndarray, np.ndarray, List[List[torch.Tensor]]]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    terms_torch = torch.as_tensor(terms_np, dtype=torch.bool, device=device)

    # policy gradient training
    pg_loss, entropy, past_grads[0] = policy_gradient_ogd(policy_net, critic_1_local, critic_2_local, obs_torch,
                                                          entropy_factor, past_grads[0], ogd_alpha, ogd_n)

    # value training
    v_loss_1, v_loss_2, q_target, q_local_1, q_local_2, past_grads[1], past_grads[2] = \
        value_train_ogd(policy_net, critic_1_target, critic_2_target, critic_1_local, critic_2_local, net_loss,
                        obs_torch, next_obs_torch, rewards_torch, terms_torch, actions_torch, gamma, entropy_factor,
                        past_grads[1], past_grads[2], ogd_alpha, ogd_n)

    soft_copy(critic_1_target, critic_1_local, tau)
    soft_copy(critic_2_target, critic_2_local, tau)

    return pg_loss, v_loss_1, v_loss_2, entropy, q_target, q_local_1, q_local_2, past_grads
