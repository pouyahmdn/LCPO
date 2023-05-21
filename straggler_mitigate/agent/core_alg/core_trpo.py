import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple, Callable
import torch.nn.functional as tfunctional
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import value_train, cumulative_rewards, gae_advantage
from agent.core_alg.core_utils import get_flat_params_from, set_flat_params_to
from neural_net.nn_perm import PermInvNet
from param import config


def conjugate_gradients(q_d_k_func: Callable[[torch.Tensor], torch.Tensor], grad_pg: torch.Tensor, nsteps: int,
                        residual_tol: float = 1e-10) -> torch.Tensor:
    x_k = torch.zeros(grad_pg.size())
    resid = grad_pg.clone()
    d_k = grad_pg.clone()
    r_dot_r = torch.dot(resid, resid)
    for i in range(nsteps):
        _q_dk = q_d_k_func(d_k)
        alpha = r_dot_r / torch.dot(d_k, _q_dk)
        x_k += alpha * d_k
        resid -= alpha * _q_dk
        new_r_dot_r = torch.dot(resid, resid)
        beta = new_r_dot_r / r_dot_r
        d_k = resid + beta * d_k
        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x_k


def linesearch(model: PermInvNet,
               loss_func: Callable[..., torch.Tensor],
               kl_func: Callable[[], torch.Tensor],
               max_kl: float,
               x_old: torch.Tensor,
               fullstep: torch.Tensor,
               expected_improve_rate,
               max_backtracks: int = 10,
               accept_ratio: float = .1) -> Tuple[bool, torch.Tensor]:
    fval = loss_func(True).data
    for _n_backtracks, stepfrac in enumerate(.5**np.arange(max_backtracks)):
        x_new = x_old + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        newfval = loss_func(True).data
        with torch.no_grad():
            new_kl = kl_func().mean()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0 and new_kl <= max_kl * 1.5:
            return True, x_new
    return False, x_old


def trpo_step(model: PermInvNet, get_loss: Callable[..., torch.Tensor], get_kl: Callable[[], torch.Tensor],
              max_kl: float, damping: float) -> Tuple[float, float, float, float]:
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def q_dot_v(v) -> torch.Tensor:
        kl = get_kl().mean()

        grads_lvl1 = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads_lvl1])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grad_lvl2 = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grad_lvl2]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(q_dot_v, -loss_grad, 15)
    v = -(loss_grad * stepdir).sum()
    if v == 0:
        lm = 0
        lr_approx = 0
    else:
        lm = torch.sqrt(v / 2 / max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = (-loss_grad * fullstep).sum(0, keepdim=True)

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, get_kl, max_kl, prev_params, fullstep, neggdotstepdir)
        set_flat_params_to(model, new_params)

        lr_approx = torch.abs(new_params-prev_params).mean()

    return loss.item(), lm, loss_grad.norm(), lr_approx


def original_trpo(policy_net: PermInvNet or torch.nn.Module, states_torch: torch.Tensor,
                  actions_torch: torch.Tensor, adv_torch: torch.Tensor, mask: torch.Tensor,
                  entropy_factor: float, monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    if mask.sum() == mask.shape[0]:
        return 0, 0, 0
    else:
        keep_mask = (1 - mask).squeeze().bool()
        actions_torch = actions_torch[keep_mask]
        adv_torch = adv_torch[keep_mask]
        states_torch = states_torch[keep_mask]

    with torch.no_grad():
        log_pi_before = tfunctional.log_softmax(policy_net.forward(states_torch), dim=-1)
        pi_before = torch.exp(log_pi_before)
        log_pi_acts_before = log_pi_before.gather(1, actions_torch)
        entropy_before = (log_pi_before * pi_before).sum(dim=-1).mean()
        log_pi_min = log_pi_acts_before.min().detach().item()

    def get_loss_trpo(volatile: bool = False) -> torch.Tensor:
        if volatile:
            with torch.no_grad():
                q = policy_net.forward(states_torch)
                log_pi = tfunctional.log_softmax(q, dim=-1)
                log_pi_acts = log_pi.gather(1, actions_torch)
                pi = torch.exp(log_pi)
                entropy = (log_pi * pi).sum(dim=-1).mean()
        else:
            q = policy_net.forward(states_torch)
            log_pi = tfunctional.log_softmax(q, dim=-1)
            log_pi_acts = log_pi.gather(1, actions_torch)
            pi = torch.exp(log_pi)
            entropy = (log_pi * pi).sum(dim=-1).mean()

        action_loss = -adv_torch * torch.exp(log_pi_acts - log_pi_acts_before)
        return action_loss.mean() + entropy * entropy_factor

    def get_kl_trpo() -> torch.Tensor:
        log_pi = tfunctional.log_softmax(policy_net.forward(states_torch), dim=-1)
        kl = (pi_before * (log_pi_before - log_pi)).sum(dim=-1)
        return kl

    trpo_loss, lm, pg_norm, lr = trpo_step(policy_net, get_loss_trpo, get_kl_trpo, max_kl=config.trpo_kl_in,
                                           damping=config.trpo_damping)
    monitor.add_scalar('lpg/lm', lm, it)
    monitor.add_scalar('lpg/pg_norm', pg_norm, it)
    monitor.add_scalar('lpg/lr_approx', lr, it)

    monitor.add_scalar('lpg/kl_in_d', get_kl_trpo().mean().item(), it)

    return trpo_loss, entropy_before, log_pi_min


def train_trpo(value_net: PermInvNet or torch.nn.Module, policy_net: PermInvNet or torch.nn.Module,
               net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
               actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray, obs_np: np.ndarray,
               dones_np: np.ndarray, masks_np: np.ndarray, gamma_rate: float, entropy_factor: float,
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
    returns_np = cumulative_rewards(rewards_np, dones_np, next_values_np[-1], gamma_rate, times_np)
    returns_torch = torch.as_tensor(returns_np, dtype=torch.float, device=device)

    # compute advantage
    adv_np = gae_advantage(rewards_np, dones_np, values_np, next_values_np, gamma_rate, times_np)
    adv_torch = torch.as_tensor(adv_np, dtype=torch.float, device=device)

    pg_loss, entropy, log_pi_min = original_trpo(policy_net, obs_torch, actions_torch, adv_torch, masks_torch,
                                                 entropy_factor, monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
