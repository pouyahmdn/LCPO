from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as tfunctional

from param import config
from neural_net.nn_perm import PermInvNet


def soft_copy(critic_1_target: PermInvNet, critic_2_target: PermInvNet, critic_1_local: PermInvNet,
              critic_2_local: PermInvNet, tau: float):
    assert 0 <= tau <= 1
    for target_nn, source_nn in [(critic_1_target, critic_1_local),
                                 (critic_2_target, critic_2_local)]:
        for target_param, source_param in zip(target_nn.parameters(), source_nn.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def policy_gradient(policy_net: PermInvNet, net_opt_p: torch.optim.Optimizer, critic_1_local: PermInvNet,
                    critic_2_local: PermInvNet, states_torch: torch.Tensor, entropy_factor: float,
                    mask_torch: torch.Tensor) -> Tuple[float, float]:
    q = policy_net.forward(states_torch)
    log_pi = tfunctional.log_softmax(q, dim=-1)
    pi = torch.exp(log_pi)

    with torch.no_grad():
        q_local_1 = critic_1_local(states_torch)
        q_local_2 = critic_2_local(states_torch)
        min_q_local = torch.min(q_local_1, q_local_2)
    loss = ((log_pi * entropy_factor - min_q_local) * pi).sum(dim=-1)[mask_torch == 0].mean()
    entropy = (log_pi * pi).sum(dim=-1)[mask_torch == 0].mean()
    assert torch.equal(mask_torch, states_torch[:, -1])

    net_opt_p.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    net_opt_p.step()

    return loss.item(), entropy.item()


def value_train(policy_net: PermInvNet, critic_1_target: PermInvNet, critic_2_target: PermInvNet,
                critic_1_local: PermInvNet, critic_2_local: PermInvNet, net_opt_v_1: torch.optim.Optimizer,
                net_opt_v_2: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, states_torch: torch.Tensor,
                next_states_torch: torch.Tensor, rewards_torch: torch.Tensor, dones_torch: torch.Tensor,
                actions_torch: torch.Tensor, gamma_rate: float, entropy_factor: float, mask_next_torch: torch.Tensor,
                mask_choice: int,  times_torch: torch.Tensor = None) -> Tuple[float, float, float, float, float]:
    with torch.no_grad():
        q = policy_net.forward(next_states_torch)
        log_pi = tfunctional.log_softmax(q, dim=-1)
        pi = torch.exp(log_pi)
        pi_mask = torch.zeros_like(pi)
        pi_mask[:, mask_choice] = 1
        next_q_target1 = critic_1_target.forward(next_states_torch)
        next_q_target2 = critic_2_target.forward(next_states_torch)
        min_qs_local = torch.min(next_q_target1, next_q_target2)
        min_next_q_target = pi * (min_qs_local - entropy_factor * log_pi)
        min_next_q_target[mask_next_torch] = 0
        min_next_q_target[mask_next_torch, mask_choice] = min_qs_local[mask_next_torch, mask_choice]
        assert torch.equal(mask_next_torch, next_states_torch[:, -1].bool())
        min_next_q_target = min_next_q_target.sum(dim=1)
        if times_torch is None:
            q_target = rewards_torch + (~dones_torch).float() * gamma_rate * min_next_q_target
        else:
            q_target = rewards_torch + (~dones_torch).float() * torch.exp(times_torch * gamma_rate) * min_next_q_target
        q_target = q_target.unsqueeze(-1)

    q_local_1 = critic_1_local(states_torch).gather(1, actions_torch)
    q_local_2 = critic_2_local(states_torch).gather(1, actions_torch)
    v_loss_1 = net_loss(q_local_1, q_target)
    v_loss_2 = net_loss(q_local_2, q_target)
    net_opt_v_1.zero_grad()
    net_opt_v_2.zero_grad()
    v_loss_1.backward()
    v_loss_2.backward()
    torch.nn.utils.clip_grad_norm_(critic_1_local.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(critic_2_local.parameters(), 1)
    net_opt_v_1.step()
    net_opt_v_2.step()
    return v_loss_1.item(), v_loss_2.item(), q_target.mean(), q_local_1.mean(), q_local_2.mean()


def train_soft_actor_critic(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
                            dones_np: np.ndarray, policy_net: PermInvNet, critic_1_target: PermInvNet,
                            critic_2_target: PermInvNet, critic_1_local: PermInvNet, critic_2_local: PermInvNet,
                            net_opt_v_1: torch.optim.Optimizer, net_opt_v_2: torch.optim.Optimizer,
                            net_opt_p: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
                            gamma_rate: float, entropy_factor: float, mask_curr_np, mask_next_np: np.ndarray,
                            mask_choice: int, times_np: np.ndarray = None) -> Tuple[float, float, float, float, float,
                                                                                    float, float]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    mask_next_torch = torch.as_tensor(mask_next_np, dtype=torch.bool, device=device)
    mask_curr_torch = torch.as_tensor(mask_curr_np, dtype=torch.float, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    times_torch = torch.as_tensor(times_np, dtype=torch.float, device=device)
    dones_torch = torch.as_tensor(dones_np, dtype=torch.bool, device=device)

    # policy gradient training
    pg_loss, entropy = policy_gradient(policy_net, net_opt_p, critic_1_local, critic_2_local, obs_torch, entropy_factor,
                                       mask_curr_torch)

    # value training
    v_loss_1, v_loss_2, q_target, q_local_1, q_local_2 = \
        value_train(policy_net, critic_1_target, critic_2_target, critic_1_local, critic_2_local, net_opt_v_1,
                    net_opt_v_2, net_loss, obs_torch, next_obs_torch, rewards_torch, dones_torch, actions_torch,
                    gamma_rate, entropy_factor, mask_next_torch, mask_choice, times_torch)

    soft_copy(critic_1_target, critic_2_target, critic_1_local, critic_2_local, config.off_policy_tau)

    return pg_loss, v_loss_1, v_loss_2, entropy, q_target, q_local_1, q_local_2


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
