import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple, Callable
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_pg import value_train, cumulative_rewards, gae_advantage, policy_gradient
from agent.core_alg.core_utils import get_flat_params_from, set_flat_params_to
from agent.core_alg.core_trpo import conjugate_gradients
from neural_net.nn import FullyConnectNN, FCNPolicy


def get_qu(a, b, c):
    sqr = np.sqrt(b ** 2 - 4 * a * c)
    return (-b + sqr) / 2 / a, (-b - sqr) / 2 / a


def linesearch(model: FCNPolicy,
               loss_func: Callable[..., torch.Tensor],
               kl_out_func: Callable[[], torch.Tensor],
               kl_in_func: Callable[[], torch.Tensor],
               max_kl_out: float,
               max_kl_in: float,
               x_old: torch.Tensor,
               fullstep: torch.Tensor,
               expected_improve_rate: torch.Tensor,
               max_backtracks: int = 10,
               accept_ratio: float = .1) -> Tuple[bool, torch.Tensor]:
    fval = loss_func(True).data
    # print("fval before", fval.item())
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
        # print("a/e/r/kli/klo", actual_improve.item(), expected_improve.item(), ratio.item(),
        #       new_kl_in.item(), new_kl_out.item())

        if ratio.item() > accept_ratio and \
                actual_improve.item() > 0 and \
                new_kl_out <= max_kl_out and \
                new_kl_in <= max_kl_in:
            # print("fval after", newfval.item())
            return True, x_new
    return False, x_old


def trpo_step(model: FCNPolicy, get_loss: Callable[..., torch.Tensor], get_kl_out: Callable[[], torch.Tensor],
              get_kl_in: Callable[[], torch.Tensor], max_kl_out: float, max_kl_in: float,
              damping: float, solve_dual: bool) -> Tuple[float, float, float, float, float]:
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

    def q_in_dot_v(v) -> torch.Tensor:
        kl_in = get_kl_in()
        kl = kl_in.mean()

        grads_lvl1 = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads_lvl1])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grad_lvl2 = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grad_lvl2]).data

        return flat_grad_grad_kl + v * damping

    stepoutdir = conjugate_gradients(q_out_dot_v, -loss_grad, 15)
    vout = -(loss_grad * stepoutdir).sum().item()
    if vout == 0:
        fullstep = torch.zeros_like(stepoutdir)
        lm = 0
    else:
        if solve_dual:
            stepindir = conjugate_gradients(q_in_dot_v, -loss_grad, 15)
            vin = -(loss_grad * stepindir).sum().item()

            a_in = [0.5 * vin, vout, 0.5 * (stepoutdir * q_in_dot_v(stepoutdir)).sum().item()]
            a_out = [0.5 * (stepindir * q_out_dot_v(stepoutdir)).sum().item(), vin, 0.5 * vout]

            diff = np.array(a_in)/max_kl_in - np.array(a_out)/max_kl_out
            diff = diff / diff[0]

            if diff[1] ** 2 <= 4 * diff[2] or max(get_qu(*diff)) <= 0:
                l_out = np.sqrt(2 * max_kl_out / vout)
                fullstep = l_out * stepoutdir
                lm = 0
            else:
                # TODO: choose s by the optimization objective
                s = max(get_qu(*diff))
                l_out = np.sqrt(max_kl_in / (a_in[0] * s ** 2 + a_in[1] * s + a_in[0]))
                l_in = l_out * s
                fullstep = l_in * stepindir + l_out * stepoutdir
                lm = s
        else:
            l_out = np.sqrt(2 * max_kl_out / vout)
            fullstep = l_out * stepoutdir
            lm = 0

    # shs_ood = 0.5 * (stepdir * q_dot_v(stepdir)).sum()
    # lm = torch.sqrt(shs_ood)
    # shs_ind = 0.5 * (stepdir * q_in_dot_v(stepdir)).sum()
    # lm = max(torch.sqrt(shs_ood / max_kl), torch.sqrt(shs_ind / max_kl_in))
    # if lm > 100:
    #     import pdb
    #     pdb.set_trace()
    # fullstep = stepdir / lm

    if vout == 0:
        theta_dir = 0
        lr_approx = 0
    else:
        neggdotstepdir = (-loss_grad * fullstep).sum(0, keepdim=True)
        theta_dir = (loss_grad * fullstep).sum()/loss_grad.norm()/fullstep.norm() / np.pi * 180

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, get_kl_out, get_kl_in, max_kl_out, max_kl_in, prev_params,
                                         fullstep, neggdotstepdir)
        set_flat_params_to(model, new_params)

        lr_approx = torch.abs(new_params-prev_params).mean()

    return loss.item(), lm, loss_grad.norm(), theta_dir, lr_approx


def locopo(policy_net: FCNPolicy, states_local_torch: torch.Tensor, actions_torch: torch.Tensor,
           adv_torch: torch.Tensor, entropy_factor: float, trpo_kl_in: float, trpo_kl_out: float, trpo_damping: float,
           trpo_dual: bool, states_global_torch: torch.Tensor, monitor: SummaryWriter, it: int) -> Tuple[float, float,
                                                                                                         float]:

    with torch.no_grad():
        log_pi_global_before = policy_net.log_pi(states_global_torch)
        pi_global_before = torch.exp(log_pi_global_before)

        log_pi_act_local_before, entropy_local_before, log_pi_local_before, \
            pi_local_before = policy_net.full_act(states_local_torch, actions_torch)
        entropy_local_before = entropy_local_before.mean()
        log_pi_min = log_pi_act_local_before.min().detach().item()

    def get_loss_lcpo(volatile: bool = False) -> torch.Tensor:
        if volatile:
            with torch.no_grad():
                log_pi_act, entropy, _, _ = policy_net.full_act(states_local_torch, actions_torch)
        else:
            log_pi_act, entropy, _, _ = policy_net.full_act(states_local_torch, actions_torch)

        action_loss = -adv_torch * torch.exp(log_pi_act - log_pi_act_local_before)
        return action_loss.mean() - entropy.mean() * entropy_factor

    def get_kl_out() -> torch.Tensor:
        log_pi_new = policy_net.log_pi(states_global_torch)
        kl = (pi_global_before * (log_pi_global_before - log_pi_new)).sum(dim=(-1, -2))
        return kl

    def get_kl_in() -> torch.Tensor:
        log_pi_new = policy_net.log_pi(states_local_torch)
        kl = (pi_local_before * (log_pi_local_before - log_pi_new)).sum(dim=(-1, -2))
        return kl

    trpo_loss, lm, pg_norm, theta_dir, lr = trpo_step(policy_net, get_loss_lcpo, get_kl_out, get_kl_in,
                                                      max_kl_out=trpo_kl_out, max_kl_in=trpo_kl_in,
                                                      damping=trpo_damping, solve_dual=trpo_dual)
    monitor.add_scalar('lpg/lm', lm, it)
    monitor.add_scalar('lpg/pg_norm', pg_norm, it)
    monitor.add_scalar('lpg/theta_dir', theta_dir, it)
    monitor.add_scalar('lpg/lr_approx', lr, it)

    monitor.add_scalar('lpg/kl_out_of_d', get_kl_out().mean().item(), it)
    monitor.add_scalar('lpg/kl_in_d', get_kl_in().mean().item(), it)

    return trpo_loss, entropy_local_before, log_pi_min


def train_lcpo(value_net: FullyConnectNN, policy_net: FCNPolicy, net_opt_p: torch.optim.Optimizer,
               net_opt_v: torch.optim.Optimizer, net_loss: torch.nn.MSELoss, device: torch.device,
               actions_np: np.ndarray, next_obs_np: np.ndarray, rewards_np: np.ndarray,
               obs_np: np.ndarray, terms_np: np.ndarray, truncs_np: np.ndarray, gamma: float,
               lam: float, trpo_kl_in: float, trpo_kl_out: float, trpo_damping: float, trpo_dual: bool,
               entropy_factor: float, ood_obs_np: np.ndarray, monitor: SummaryWriter,
               it: int) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, np.ndarray]:

    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)
    ood_obs_torch = torch.as_tensor(ood_obs_np, dtype=torch.float, device=device)

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

    if len(ood_obs_np) > 0:
        # locally constrained policy optimization
        pg_loss, entropy, log_pi_min = locopo(policy_net, obs_torch, actions_torch, adv_torch, entropy_factor,
                                              trpo_kl_in, trpo_kl_out, trpo_damping, trpo_dual, ood_obs_torch, monitor,
                                              it)
    else:
        # advantage actor critic
        pg_loss, entropy, log_pi_min = \
            policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, entropy_factor, monitor, it)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, log_pi_min, adv_np
