import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple, Callable
import torch.nn.functional as tfunctional
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import value_train, cumulative_rewards, gae_advantage, policy_gradient
from agent.core_alg.core_utils import get_flat_params_from, set_flat_params_to
from agent.core_alg.core_trpo import conjugate_gradients
from neural_net.nn_perm import PermInvNet
from param import config


def linesearch(model: PermInvNet,
               loss_func: Callable[..., torch.Tensor],
               kl_out_func: Callable[[], torch.Tensor],
               kl_in_func: Callable[[], torch.Tensor],
               max_kl_out: float,
               max_kl_in: float,
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
            new_kl_in = kl_in_func().mean()
            new_kl_out = kl_out_func().mean()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0 and new_kl_out <= max_kl_out and new_kl_in <= max_kl_in:
            return True, x_new
    return False, x_old


def trpo_step(model: PermInvNet, get_loss: Callable[..., torch.Tensor], get_kl_out: Callable[[], torch.Tensor],
              get_kl_in: Callable[[], torch.Tensor], max_kl_out: float, max_kl_in: float,
              damping: float) -> Tuple[float, float, float, float, float]:
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def q_out_dot_v(v) -> torch.Tensor:
        kl_out = get_kl_out()
        kl = kl_out.mean()

        grads_lvl1 = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads_lvl1])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grad_lvl2 = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grad_lvl2]).data

        return flat_grad_grad_kl + v * damping

    stepoutdir = conjugate_gradients(q_out_dot_v, -loss_grad, 15)
    vout = -(loss_grad * stepoutdir).sum().item()
    if vout == 0:
        lm = 0
        theta_dir = 0
        lr_approx = 0
    else:
        l_out = np.sqrt(2 * max_kl_out / vout)
        fullstep = l_out * stepoutdir
        lm = 0
        neggdotstepdir = (-loss_grad * fullstep).sum(0, keepdim=True)
        theta_dir = (loss_grad * fullstep).sum()/loss_grad.norm()/fullstep.norm() / np.pi * 180

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, get_kl_out, get_kl_in, max_kl_out, max_kl_in, prev_params,
                                         fullstep, neggdotstepdir)
        set_flat_params_to(model, new_params)

        lr_approx = torch.abs(new_params-prev_params).mean()

    return loss.item(), lm, loss_grad.norm(), theta_dir, lr_approx


def local_trpo(policy_net: PermInvNet or torch.nn.Module, states_local_torch: torch.Tensor,
               actions_torch: torch.Tensor, adv_torch: torch.Tensor, mask: torch.Tensor,
               entropy_factor: float, states_global_torch: torch.Tensor,
               monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    if mask.sum() == mask.shape[0]:
        return 0, 0, 0
    else:
        keep_mask = (1 - mask).squeeze().bool()
        states_local_torch = states_local_torch[keep_mask]
        actions_torch = actions_torch[keep_mask]
        adv_torch = adv_torch[keep_mask]

    with torch.no_grad():
        log_pi_global_before = tfunctional.log_softmax(policy_net.forward(states_global_torch), dim=-1)
        pi_global_before = torch.exp(log_pi_global_before)

        log_pi_local_before = tfunctional.log_softmax(policy_net.forward(states_local_torch), dim=-1)
        pi_local_before = torch.exp(log_pi_local_before)
        log_pi_acts_local_before = log_pi_local_before.gather(1, actions_torch)
        entropy_local_before = (log_pi_local_before * pi_local_before).sum(dim=-1).mean()
        log_pi_min = log_pi_acts_local_before.min().detach().item()

    def get_loss_trpo(volatile: bool = False) -> torch.Tensor:
        if volatile:
            with torch.no_grad():
                q_local = policy_net.forward(states_local_torch)
                log_pi = tfunctional.log_softmax(q_local, dim=-1)
                log_pi_acts = log_pi.gather(1, actions_torch)
                pi = torch.exp(log_pi)
                entropy = (log_pi * pi).sum(dim=-1).mean()
        else:
            q_local = policy_net.forward(states_local_torch)
            log_pi = tfunctional.log_softmax(q_local, dim=-1)
            log_pi_acts = log_pi.gather(1, actions_torch)
            pi = torch.exp(log_pi)
            entropy = (log_pi * pi).sum(dim=-1).mean()

        action_loss = -adv_torch * torch.exp(log_pi_acts - log_pi_acts_local_before)
        return action_loss.mean() + entropy * entropy_factor

    def get_kl_trpo() -> torch.Tensor:
        log_pi_global = tfunctional.log_softmax(policy_net.forward(states_global_torch), dim=-1)
        kl = (pi_global_before * (log_pi_global_before - log_pi_global)).sum(dim=-1)
        return kl

    def get_kl_in_trpo() -> torch.Tensor:
        log_pi_local = tfunctional.log_softmax(policy_net.forward(states_local_torch), dim=-1)
        kl = (pi_local_before * (log_pi_local_before - log_pi_local)).sum(dim=-1)
        return kl

    trpo_loss, lm, pg_norm, theta_dir, lr = trpo_step(policy_net, get_loss_trpo, get_kl_trpo, get_kl_in_trpo,
                                                      max_kl_out=config.trpo_kl_out, max_kl_in=config.trpo_kl_in,
                                                      damping=config.trpo_damping)
    monitor.add_scalar('lpg/lm', lm, it)
    monitor.add_scalar('lpg/pg_norm', pg_norm, it)
    monitor.add_scalar('lpg/theta_dir', theta_dir, it)
    monitor.add_scalar('lpg/lr_approx', lr, it)

    monitor.add_scalar('lpg/kl_out_of_d', get_kl_trpo().mean().item(), it)
    monitor.add_scalar('lpg/kl_in_d', get_kl_in_trpo().mean().item(), it)

    return trpo_loss, entropy_local_before, log_pi_min


def train_trpo(value_net: PermInvNet or torch.nn.Module, policy_net: PermInvNet or torch.nn.Module,
               net_opt_p: torch.optim.Optimizer, net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss,
               device: torch.device, actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray,
               obs_np: np.ndarray, dones_np: np.ndarray, masks_np: np.ndarray, gamma_rate: float,
               entropy_factor: float, ood_obs_np: np.ndarray, monitor: SummaryWriter, it: int,
               times_np: np.ndarray = None) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, np.ndarray]:

    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    masks_torch = torch.as_tensor(masks_np, dtype=torch.float, device=device)
    ood_obs_torch = torch.as_tensor(ood_obs_np, dtype=torch.float, device=device)

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

    if len(ood_obs_np) > 0:
        # local trust region policy optimization
        pg_loss, entropy, log_pi_min = local_trpo(policy_net, obs_torch, actions_torch, adv_torch, masks_torch,
                                                  entropy_factor, ood_obs_torch, monitor, it)
    else:
        # advantage actor critic
        pg_loss, entropy, log_pi_min = policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch,
                                                             masks_torch, entropy_factor, monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
