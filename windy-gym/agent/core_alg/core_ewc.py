from typing import Tuple, Dict, List
import numpy as np
import torch

from .core_dqn import soft_copy
from neural_net.nn import FCNPolicy


def policy_gradient_ewc(policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer, critic_1_local: FCNPolicy,
                        critic_2_local: FCNPolicy, states_torch: torch.Tensor, entropy_factor: float,
                        importances: Dict[str, torch.Tensor], past_weights: Dict[str, torch.Tensor],
                        ewc_alpha: float, ewc_gamma: float) -> Tuple[
    float, float, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

    for n, p in policy_net.named_parameters():
        if p.requires_grad and n in importances:
            theta_past = past_weights[n]
            fisher = ewc_gamma * importances[n]
            loss += 0.5 * ewc_alpha * (fisher * (p - theta_past) ** 2).sum()

    net_opt_p.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    net_opt_p.step()

    # No task labels, so use pi itself as a task label, and the FIM turns to expectation of gradient squares with random
    # actions sampled from pi and backpropagation through log_pi
    importances_new = {}
    for i in range(states_torch.shape[0]):
        policy_net.zero_grad()
        _, _log_pi_act = policy_net.sample_action_prob(states_torch[i])
        assert len(_log_pi_act.shape) == 1, _log_pi_act.shape
        _log_pi_act.sum(-1).backward()
        for n, p in policy_net.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if n not in importances_new:
                        importances_new[n] = p.grad.detach().clone() ** 2
                    else:
                        importances_new[n] += p.grad.detach().clone() ** 2

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

    return loss.item(), entropy.mean().item(), importances, past_weights


def value_train_ewc(policy_net: FCNPolicy, critic_1_target: FCNPolicy, critic_2_target: FCNPolicy,
                    critic_1_local: FCNPolicy, critic_2_local: FCNPolicy, net_opt_v_1: torch.optim.Optimizer,
                    net_opt_v_2: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, states_torch: torch.Tensor,
                    next_states_torch: torch.Tensor, rewards_torch: torch.Tensor, terms_torch: torch.Tensor,
                    actions_torch: torch.Tensor, gamma: float, entropy_factor: float,
                    importances_1: Dict[str, torch.Tensor], past_weights_1: Dict[str, torch.Tensor],
                    importances_2: Dict[str, torch.Tensor], past_weights_2: Dict[str, torch.Tensor],
                    ewc_alpha: float, ewc_gamma: float) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray,
Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

    for n, p in critic_1_local.named_parameters():
        if p.requires_grad and n in importances_1:
            theta_past = past_weights_1[n]
            fisher = ewc_gamma * importances_1[n]
            v_loss_1 += 0.5 * ewc_alpha * (fisher * (p - theta_past) ** 2).sum()

    net_opt_v_1.zero_grad()
    v_loss_1.backward()
    torch.nn.utils.clip_grad_norm_(critic_1_local.parameters(), 1)
    net_opt_v_1.step()

    v_loss_2 = net_loss(q_local_2, q_target)

    for n, p in critic_2_local.named_parameters():
        if p.requires_grad and n in importances_2:
            theta_past = past_weights_2[n]
            fisher = ewc_gamma * importances_2[n]
            v_loss_1 += 0.5 * ewc_alpha * (fisher * (p - theta_past) ** 2).sum()

    net_opt_v_2.zero_grad()
    v_loss_2.backward()
    torch.nn.utils.clip_grad_norm_(critic_2_local.parameters(), 1)
    net_opt_v_2.step()

    critic_1_local.zero_grad()
    # Use MSE loss instead of NLL, as the network predicts Q-values, not probabilities

    importances_new = {}
    for i in range(states_torch.shape[0]):
        policy_net.zero_grad()
        _q_local_1 = critic_1_local(states_torch[i])[0].gather(-1, actions_torch[i].unsqueeze(dim=-1)).squeeze(dim=-1).sum(
            dim=-1)
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
    for i in range(states_torch.shape[0]):
        policy_net.zero_grad()
        _q_local_2 = critic_2_local(states_torch[i])[0].gather(-1, actions_torch[i].unsqueeze(dim=-1)).squeeze(dim=-1).sum(dim=-1)
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

    return v_loss_1.item(), v_loss_2.item(), q_target.detach().numpy(), q_local_1.detach().numpy(), \
        q_local_2.detach().numpy(), importances_1, past_weights_1, importances_2, past_weights_2


def train_sac_ewc(actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
                  terms_np: np.ndarray, truncs_np: np.ndarray, policy_net: FCNPolicy, critic_1_target: FCNPolicy,
                  critic_2_target: FCNPolicy, critic_1_local: FCNPolicy, critic_2_local: FCNPolicy,
                  net_opt_v_1: torch.optim.Optimizer, net_opt_v_2: torch.optim.Optimizer,
                  net_opt_p: torch.optim.Optimizer,
                  net_loss: torch.nn.MSELoss, device: torch.device, gamma: float, entropy_factor: float,
                  tau: float, importances: List[Dict[str, torch.Tensor]],
                  past_weights: List[Dict[str, torch.Tensor]], ewc_alpha: float, ewc_gamma: float) -> Tuple[
    float, float, float, float, np.ndarray,
    np.ndarray, np.ndarray, List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    next_obs_torch = torch.as_tensor(next_obs_np, dtype=torch.float, device=device)
    rewards_torch = torch.as_tensor(rewards_np, dtype=torch.float, device=device)
    terms_torch = torch.as_tensor(terms_np, dtype=torch.bool, device=device)

    # policy gradient training
    pg_loss, entropy, importances[0], past_weights[0] = policy_gradient_ewc(policy_net, net_opt_p, critic_1_local,
                                                                            critic_2_local, obs_torch, entropy_factor,
                                                                            importances[0], past_weights[0], ewc_alpha,
                                                                            ewc_gamma)

    # value training
    v_loss_1, v_loss_2, q_target, q_local_1, q_local_2, importances[1], past_weights[1], importances[2], past_weights[
        2] = \
        value_train_ewc(policy_net, critic_1_target, critic_2_target, critic_1_local, critic_2_local, net_opt_v_1,
                        net_opt_v_2, net_loss, obs_torch, next_obs_torch, rewards_torch, terms_torch, actions_torch,
                        gamma, entropy_factor, importances[1], past_weights[1], importances[2], past_weights[2],
                        ewc_alpha, ewc_gamma)

    soft_copy(critic_1_target, critic_1_local, tau)
    soft_copy(critic_2_target, critic_2_local, tau)

    return pg_loss, v_loss_1, v_loss_2, entropy, q_target, q_local_1, q_local_2, importances, past_weights
