import numpy as np
from param import config


def safe_condition(obs):
    # max queue size exceeds threshold
    return np.max(obs[:config.num_servers]) > config.tw_safe_queue_size
