from typing import List
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as TrainerNetA2C
from env.grid import ShakyGrid
from neural_net.tab import TabularPolicy
from neural_net.nn import FullyConnectNN
from buffer.buffer import TransitionBuffer
from utils.rms import RunningMeanStd


class TrainerNet(TrainerNetA2C):
    def __init__(self, environment: ShakyGrid, environment_eval: ShakyGrid, monitor: SummaryWriter, output_folder: str,
                 device: str, seed: int, nn_hids: List[int], batch_size: int, reward_scale: float, entropy_max: float,
                 entropy_min: float, entropy_decay: float, lr_rate: float, val_lr_rate: float, gamma: float, lam: float,
                 auto_target_entropy: float, ent_lr: float, len_buff_eval: int):
        super(TrainerNet, self).__init__(environment, environment_eval, monitor, output_folder, device, seed, nn_hids,
                                         batch_size, reward_scale, entropy_max, entropy_min, entropy_decay, lr_rate,
                                         val_lr_rate, gamma, lam, auto_target_entropy, ent_lr, len_buff_eval)
        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=lr_rate, weight_decay=0, eps=1e-5)

        self.device = torch.device(device)
        if device != "cpu":
            torch.cuda.set_device(self.device)
        # CUDA seed affects NN initial weights and policy network decisions
        self.seed = seed
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        self.obs_len = environment.obs_dim
        self.act_len = environment.action_dim
        self.act_bins = environment.n_bins
        self.env = environment
        self.env_eval = environment_eval
        self.monitor = monitor
        self.output_folder = output_folder

        self.policy_net = TabularPolicy(self.obs_len, nn_hids, self.act_len * self.act_bins, self.act_len, act=nn.ReLU,
                                        final_layer_act=False)
        self.value_net = FullyConnectNN(self.obs_len, nn_hids, 1, 1, act=nn.ReLU, final_layer_act=False)
        self.value_net = torch.jit.script(self.value_net).to(self.device)

        self.batch_size = batch_size
        self.buff = TransitionBuffer(self.obs_len, self.act_len, batch_size, reward_scale)
        self.buff_eval = TransitionBuffer(self.obs_len, self.act_len, len_buff_eval, reward_scale)

        self.entropy_factor = entropy_max
        self.entropy_max = entropy_max
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min

        self.net_opt_p = torch.optim.Adam(self.policy_net.parameters(), lr=lr_rate, weight_decay=0, eps=1e-5)
        self.net_opt_p_lr = lr_rate
        self.net_opt_v = torch.optim.Adam(self.value_net.parameters(), lr=val_lr_rate, weight_decay=1e-4, eps=1e-5)

        self.it = 0

        self.maximum_entropy = self.act_len * np.log(self.act_bins)
        self.minimum_entropy = 0

        self.auto_target_entropy = auto_target_entropy
        if auto_target_entropy > 0:
            self.log_entropy = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_entropy = self.maximum_entropy * auto_target_entropy
            self.opt_ent = torch.optim.Adam([self.log_entropy], lr=ent_lr, weight_decay=1e-4)

        self.net_loss = torch.nn.MSELoss(reduction='mean')
        self.gamma = gamma
        self.lamb = lam

        self.ret = 0
        self.ret_list = []
        self.ret_rms = RunningMeanStd(shape=())
