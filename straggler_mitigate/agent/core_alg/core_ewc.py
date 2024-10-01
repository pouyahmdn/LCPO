import numpy as np
import torch
import torch.nn.functional as tfunctional

from typing import List, Dict, Tuple
from param import config
from neural_net.nn_perm import PermInvNet
from agent.core_alg.core_sac import soft_copy


def policy_gradient_ewc(policy_net: PermInvNet, net_opt_p: torch.optim.Optimizer, critic_1_local: PermInvNet,
                        critic_2_local: PermInvNet, states_torch: torch.Tensor, entropy_factor: float,
                        mask_torch: torch.Tensor, importances: Dict[str, torch.Tensor],
                        past_weights: Dict[str, torch.Tensor], ewc_alpha: float, ewc_gamma: float) -> Tuple[
    float, float, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

    for n, p in policy_net.named_parameters():
        if p.requires_grad and n in importances:
            theta_past = past_weights[n]
            fisher = ewc_gamma * importances[n]
            loss += 0.5 * ewc_alpha * (fisher * (p - theta_past) ** 2).sum()

    net_opt_p.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    net_opt_p.step()

    # No task labels, so use pi itself as a task label, and the NLL turns to entropy
    importances_new = {}
    for i in range(min(states_torch.shape[0], 32)):
        if mask_torch[i] != 0:
            continue
        policy_net.zero_grad()
        _q = policy_net.forward(states_torch[i])
        _log_pi = tfunctional.log_softmax(_q, dim=-1)
        with torch.no_grad():
            _pi = torch.exp(_log_pi).cpu()
            _act = (_pi[:-1].cumsum(-1) <= torch.rand(1)).sum().item()
        _log_pi[_act].backward()
        for n, p in policy_net.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if n not in importances_new:
                        importances_new[n] = p.grad.detach().clone() ** 2
                    else:
                        importances_new[n] += p.grad.detach().clone() ** 2

    if len(importances_new.keys()) > 0:
        for n, p in policy_net.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if n not in importances:
                        importances[n] = importances_new[n] / states_torch.shape[0]
                        past_weights[n] = p.data.detach().clone()
                    else:
                        importances[n] = ewc_gamma * importances[n] + (1 - ewc_gamma) * importances_new[n] / \
                                         states_torch.shape[0]
                        past_weights[n] = ewc_gamma * past_weights[n] + (1 - ewc_gamma) * p.data.detach().clone()

    return loss.item(), entropy.item(), importances, past_weights


def value_train_ewc(policy_net: PermInvNet, critic_1_target: PermInvNet, critic_2_target: PermInvNet,
                    critic_1_local: PermInvNet, critic_2_local: PermInvNet, net_opt_v_1: torch.optim.Optimizer,
                    net_opt_v_2: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, states_torch: torch.Tensor,
                    next_states_torch: torch.Tensor, rewards_torch: torch.Tensor, dones_torch: torch.Tensor,
                    actions_torch: torch.Tensor, gamma_rate: float, entropy_factor: float,
                    mask_next_torch: torch.Tensor, mask_choice: int, importances_1: Dict[str, torch.Tensor],
                    past_weights_1: Dict[str, torch.Tensor], importances_2: Dict[str, torch.Tensor],
                    past_weights_2: Dict[str, torch.Tensor], ewc_alpha: float, ewc_gamma: float, times_torch=None) -> \
        Tuple[
            float, float, float, float, float, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[
                str, torch.Tensor], Dict[
                str, torch.Tensor]]:
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

    for n, p in critic_1_local.named_parameters():
        if p.requires_grad and n in importances_1:
            theta_past = past_weights_1[n]
            fisher = ewc_gamma * importances_1[n]
            v_loss_1 += 0.5 * ewc_alpha * (fisher * (p - theta_past) ** 2).sum()

    v_loss_2 = net_loss(q_local_2, q_target)

    for n, p in critic_2_local.named_parameters():
        if p.requires_grad and n in importances_2:
            theta_past = past_weights_2[n]
            fisher = ewc_gamma * importances_2[n]
            v_loss_1 += 0.5 * ewc_alpha * (fisher * (p - theta_past) ** 2).sum()

    net_opt_v_1.zero_grad()
    net_opt_v_2.zero_grad()
    v_loss_1.backward()
    v_loss_2.backward()
    torch.nn.utils.clip_grad_norm_(critic_1_local.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(critic_2_local.parameters(), 1)
    net_opt_v_1.step()
    net_opt_v_2.step()

    critic_1_local.zero_grad()
    # Use MSE loss instead of NLL, as the network predicts Q-values, not probabilities

    importances_new = {}
    for i in range(min(states_torch.shape[0], 32)):
        policy_net.zero_grad()
        _q_local_1 = critic_1_local(states_torch[i]).gather(0, actions_torch[i])[0]
        ((_q_local_1 - q_target[i]) ** 2).backward()
        for n, p in critic_1_local.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if n not in importances_new:
                        importances_new[n] = p.grad.detach().clone() ** 2
                    else:
                        importances_new[n] += p.grad.detach().clone() ** 2

    for n, p in critic_1_local.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                if n not in importances_1:
                    importances_1[n] = importances_new[n] / states_torch.shape[0]
                    past_weights_1[n] = p.data.detach().clone()
                else:
                    importances_1[n] = ewc_gamma * importances_1[n] + (1 - ewc_gamma) * importances_new[n] / \
                                       states_torch.shape[0]
                    past_weights_1[n] = ewc_gamma * past_weights_1[n] + (1 - ewc_gamma) * p.data.detach().clone()

    critic_2_local.zero_grad()
    # Use MSE loss instead of NLL, as the network predicts Q-values, not probabilities
    importances_new = {}
    for i in range(min(states_torch.shape[0], 32)):
        policy_net.zero_grad()
        _q_local_2 = critic_2_local(states_torch[i]).gather(0, actions_torch[i])[0]
        ((_q_local_2 - q_target[i]) ** 2).backward()
        for n, p in critic_2_local.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if n not in importances_new:
                        importances_new[n] = p.grad.detach().clone() ** 2
                    else:
                        importances_new[n] += p.grad.detach().clone() ** 2

    for n, p in critic_2_local.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                if n not in importances_2:
                    importances_2[n] = importances_new[n] / states_torch.shape[0]
                    past_weights_2[n] = p.data.detach().clone()
                else:
                    importances_2[n] = ewc_gamma * importances_2[n] + (1 - ewc_gamma) * importances_new[n] / \
                                       states_torch.shape[0]
                    past_weights_2[n] = ewc_gamma * past_weights_2[n] + (1 - ewc_gamma) * p.data.detach().clone()

    return v_loss_1.item(), v_loss_2.item(), q_target.mean(), q_local_1.mean(), q_local_2.mean(), importances_1, past_weights_1, importances_2, past_weights_2


def train_sac_ewc(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
                  dones_np: np.ndarray, policy_net: PermInvNet, critic_1_target: PermInvNet,
                  critic_2_target: PermInvNet, critic_1_local: PermInvNet, critic_2_local: PermInvNet,
                  net_opt_v_1: torch.optim.Optimizer, net_opt_v_2: torch.optim.Optimizer,
                  net_opt_p: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device, gamma_rate: float,
                  entropy_factor: float, mask_curr_np, mask_next_np: np.ndarray, mask_choice: int,
                  importances: List[Dict[str, torch.Tensor]], past_weights: List[Dict[str, torch.Tensor]],
                  ewc_alpha: float, ewc_gamma: float, times_np=None) -> Tuple[float, float, float, float, float, float,
float, List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    mask_next_torch = torch.as_tensor(mask_next_np, dtype=torch.bool, device=device)
    mask_curr_torch = torch.as_tensor(mask_curr_np, dtype=torch.float, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    times_torch = torch.as_tensor(times_np, dtype=torch.float, device=device)
    dones_torch = torch.as_tensor(dones_np, dtype=torch.bool, device=device)

    # policy gradient training
    pg_loss, entropy, importances[0], past_weights[0] = policy_gradient_ewc(policy_net, net_opt_p, critic_1_local,
                                                                            critic_2_local, obs_torch, entropy_factor,
                                                                            mask_curr_torch, importances[0],
                                                                            past_weights[0], ewc_alpha, ewc_gamma)

    # value training
    v_loss_1, v_loss_2, q_target, q_local_1, q_local_2, importances[1], past_weights[1], importances[2], past_weights[
        2] = \
        value_train_ewc(policy_net, critic_1_target, critic_2_target, critic_1_local, critic_2_local, net_opt_v_1,
                        net_opt_v_2, net_loss, obs_torch, next_obs_torch, rewards_torch, dones_torch, actions_torch,
                        gamma_rate, entropy_factor, mask_next_torch, mask_choice, importances[1], past_weights[1],
                        importances[2], past_weights[2],
                        ewc_alpha, ewc_gamma, times_torch)

    soft_copy(critic_1_target, critic_2_target, critic_1_local, critic_2_local, config.off_policy_tau)

    return pg_loss, v_loss_1, v_loss_2, entropy, q_target, q_local_1, q_local_2, importances, past_weights
