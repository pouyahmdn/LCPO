from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as tfunctional

from param import config


def mlp_seq(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class PermInvNet(nn.Module):
    def __init__(self, observation_len: int, out_space: int, num_slots: int, aux_state_size: int = 0):
        super(PermInvNet, self).__init__()
        # We assume data permutation invariant data happens in batches of num_slots
        # So by indexing the slot, it would be like 0 1 2 0 1 2 ...
        self.obs_space = observation_len
        self.out_space = out_space
        self.aux_size = aux_state_size
        self.slot_size = num_slots
        assert (observation_len - aux_state_size) % num_slots == 0
        self.map_in_size = (observation_len - aux_state_size) // num_slots
        self.elem_idx = [i // self.map_in_size + (i % self.map_in_size) * self.slot_size for i in
                         range(observation_len - aux_state_size)]

        map_hid = config.nn_map
        red_hid = config.nn_red

        self.mapper = mlp_seq([self.map_in_size] + map_hid + [1],
                              activation=nn.ReLU,
                              output_activation=nn.Identity)
        self.reducer = mlp_seq([1 + self.aux_size] + red_hid + [self.out_space],
                               activation=nn.ReLU,
                               output_activation=nn.Identity)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if len(observation.shape) == 1:
            obs_inv = observation[self.elem_idx].view(self.slot_size, self.map_in_size)
        elif len(observation.shape) == 2:
            obs_inv = observation[:, self.elem_idx].view(-1, self.slot_size, self.map_in_size)
        else:
            obs_inv = observation[:, :, self.elem_idx].view(-1, self.slot_size, self.map_in_size)
        mapped_inter_values = self.mapper(obs_inv).squeeze(dim=-1)
        sum_inter = mapped_inter_values.mean(dim=-1, keepdim=True)
        if self.aux_size != 0:
            sum_inter = torch.cat((sum_inter, observation[..., -self.aux_size:]), dim=-1)
        return self.reducer(sum_inter).squeeze(dim=-1)

    @torch.jit.export
    def sample_policy(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pi = self.forward(observation)
            pi_cpu = tfunctional.softmax(pi, dim=-1).cpu()
        return pi_cpu

    @torch.jit.export
    def max(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.forward(observation)
        q_best, choice_best = torch.max(values, -1)
        return q_best.detach(), choice_best.detach()
