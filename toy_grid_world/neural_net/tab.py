from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as tfunctional

from neural_net.nn import FCNPolicy


class TabularPolicy(FCNPolicy):
    def __init__(self, in_size: int, n_hids: List[int], out_size: int, grp_size: int,
                 act: nn.modules.Module = nn.LeakyReLU, final_layer_act: bool = True):
        nn.Module.__init__(self)
        self.eps = 1e-6
        self.in_size = in_size
        assert in_size == 4
        self.out_size = out_size
        assert out_size == 4
        self.grp_size = grp_size
        assert grp_size == 1
        assert self.out_size % self.grp_size == 0
        self.bins = self.out_size // self.grp_size
        self.table = torch.nn.Parameter(torch.zeros((3, 3, 2, 2, 1, 4)))
        self.register_parameter('table', self.table)

    @torch.jit.export
    def forward(self, in_vec: torch.Tensor):
        in_vec_int = in_vec.long()
        return self.table[in_vec_int[..., 0], in_vec_int[..., 1],
                          in_vec_int[..., 2], in_vec_int[..., 3]].view((-1, self.grp_size, self.bins))

    @torch.jit.export
    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.forward(obs)
        pi_cpu = tfunctional.softmax(features, dim=-1).cpu()
        # array of shape (1, act_dim, act_bins)
        act = (pi_cpu[0, :, :-1].cumsum(-1) <= torch.rand((self.grp_size, 1))).sum(dim=-1)
        return act

    @torch.jit.export
    def max(self, in_vec: torch.Tensor):
        q_values_actions = self.forward(in_vec)
        q_best, choice_best = torch.max(q_values_actions, -1)
        return q_best.detach(), choice_best.detach()

    @torch.jit.export
    def dist(self, in_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.forward(in_vec)
        log_pi = tfunctional.log_softmax(features, dim=-1)
        pi = log_pi.exp()
        entropy = - (pi * log_pi).sum(dim=(-2, -1))

        return entropy, log_pi, pi

    @torch.jit.export
    def full_act(self, in_vec: torch.Tensor, act_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                             torch.Tensor]:
        features = self.forward(in_vec)
        log_pi = tfunctional.log_softmax(features, dim=-1)
        pi = log_pi.exp()
        log_pi_act = log_pi.gather(-1, act_vec.unsqueeze(-1)).squeeze(-1).sum(-1)
        entropy = - (pi * log_pi).sum(dim=(-2, -1))

        return log_pi_act, entropy, log_pi, pi

    @torch.jit.export
    def entropy(self, in_vec: torch.Tensor) -> torch.Tensor:
        features = self.forward(in_vec)
        log_pi = tfunctional.log_softmax(features, dim=-1)
        pi = log_pi.exp()
        entropy = - (pi * log_pi).sum(dim=(-2, -1))

        return entropy

    @torch.jit.export
    def log_pi(self, in_vec: torch.Tensor) -> torch.Tensor:
        features = self.forward(in_vec)
        log_pi = tfunctional.log_softmax(features, dim=-1)
        return log_pi

    @torch.jit.export
    def pi(self, in_vec: torch.Tensor) -> torch.Tensor:
        return self.log_pi(in_vec).exp()
