import numpy as np
import torch

from neural_net.nn import FullyConnectNN


def get_kl(log_pi_old: torch.Tensor, pi_old: torch.Tensor, log_pi_new: torch.Tensor) -> torch.Tensor:
    kl = (pi_old * (log_pi_old - log_pi_new)).sum(dim=(-1, -2))
    return kl


def get_flat_params_from(model: FullyConnectNN) -> torch.Tensor:
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model: FullyConnectNN, flat_params: torch.Tensor):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net: FullyConnectNN, grad_grad=False) -> torch.Tensor:
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad
