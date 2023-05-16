import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple, Callable
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import value_train, cumulative_rewards, gae_advantage
from agent.core_alg.core_utils import get_flat_params_from, set_flat_params_to
from neural_net.nn import FullyConnectNN, FCNPolicy


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


def linesearch(model: FullyConnectNN,
               loss_func: Callable[..., torch.Tensor],
               kl_func: Callable[[], torch.Tensor],
               max_kl: float,
               x_old: torch.Tensor,
               fullstep: torch.Tensor,
               expected_improve_rate,
               max_backtracks: int = 10,
               accept_ratio: float = .1) -> Tuple[bool, torch.Tensor]:
    fval = loss_func(True).data
    # print("fval before", fval.item())
    for _n_backtracks, stepfrac in enumerate(.5**np.arange(max_backtracks)):
        x_new = x_old + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        newfval = loss_func(True).data
        with torch.no_grad():
            new_kl = kl_func().mean()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r/kli/klo", actual_improve.item(), expected_improve.item(), ratio.item(), new_kl.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0 and new_kl <= max_kl * 1.5:
            # print("fval after", newfval.item())
            return True, x_new
    return False, x_old


def trpo_step(model: FCNPolicy, get_loss: Callable[..., torch.Tensor], get_kl: Callable[[], torch.Tensor],
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


def trpo(policy_net: FCNPolicy, states_torch: torch.Tensor,
         actions_torch: torch.Tensor, adv_torch: torch.Tensor, entropy_factor: float, trpo_kl: float,
         trpo_damping: float, monitor: SummaryWriter, it: int) -> Tuple[float, float, float]:
    with torch.no_grad():
        log_pi_act_before, entropy_before, log_pi_before, pi_before = policy_net.full_act(states_torch, actions_torch)
        entropy_before = entropy_before.mean()
        log_pi_min = log_pi_act_before.min().detach().item()

    def get_loss_trpo(volatile: bool = False) -> torch.Tensor:
        if volatile:
            with torch.no_grad():
                log_pi_act, entropy, _, _ = policy_net.full_act(states_torch, actions_torch)
        else:
            log_pi_act, entropy, _, _ = policy_net.full_act(states_torch, actions_torch)

        action_loss = -adv_torch * torch.exp(log_pi_act - log_pi_act_before)
        return action_loss.mean() - entropy.mean() * entropy_factor

    def get_kl_trpo() -> torch.Tensor:
        log_pi_new = policy_net.log_pi(states_torch)
        kl = (pi_before * (log_pi_before - log_pi_new)).sum(dim=(-1, -2))
        return kl

    trpo_loss, lm, pg_norm, lr = trpo_step(policy_net, get_loss_trpo, get_kl_trpo, max_kl=trpo_kl, damping=trpo_damping)

    monitor.add_scalar('lpg/lm', lm, it)
    monitor.add_scalar('lpg/pg_norm', pg_norm, it)
    monitor.add_scalar('lpg/lr_approx', lr, it)
    monitor.add_scalar('lpg/kl_in_d', get_kl_trpo().mean().item(), it)

    return trpo_loss, entropy_before, log_pi_min


def train_trpo(value_net: FullyConnectNN, policy_net: FCNPolicy, net_opt_v: torch.optim.Optimizer,
               net_loss: torch.nn.MSELoss, device: torch.device, actions_np: np.ndarray, next_obs_np: np.ndarray,
               rewards_np: np.ndarray, obs_np: np.ndarray, terms_np: np.ndarray, truncs_np: np.ndarray, gamma: float,
               lam: float, trpo_kl: float, trpo_damping: float, entropy_factor: float,
               monitor: SummaryWriter, it: int) -> Tuple[float, float, float, np.ndarray, np.ndarray, float,
                                                         np.ndarray]:
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

    # trust region policy optimization
    pg_loss, entropy, log_pi_min = trpo(policy_net, obs_torch, actions_torch, adv_torch, entropy_factor,
                                        trpo_kl, trpo_damping, monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
