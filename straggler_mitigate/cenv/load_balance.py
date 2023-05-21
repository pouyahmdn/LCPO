from typing import List
import numpy as np

from cenv.clb import pyenv
from param import config
from cenv.clb.pyjobgenfile import PyJobGenFile


def load_balance_env(output_folder: str) -> pyenv.PyLoadBalanceEnv:
    workload = np.load(f'{config.dataset_folder}/real_tr{config.trace_ind}.npy')
    sizes = workload[:, 0]
    arrs = workload[:, 1]

    job_gen = PyJobGenFile([sizes], [arrs], 0, 1.0e8 * 3600 * 1000, config.num_servers, config.seed, 1)

    return pyenv.PyLoadBalanceEnv(config.num_servers, config.time_window * 1000, True, False, False, True,
                                  output_folder + 'data.log',
                                  config.lb_timeout_levels, [0.75, 0.85], config.max_num_retries, True,
                                  config.tw_safe_queue_size, config.tw_exit_queue_size, job_gen, job_gen.get_ptr(),
                                  config.seed, config.skip_log)


def load_balance_env_multi(output_folder: str, skip_log: bool, num_models: int) -> List[pyenv.PyLoadBalanceEnv]:
    workload = np.load(f'{config.dataset_folder}/real_tr{config.trace_ind}.npy')
    sizes = workload[:, 0]
    arrs = workload[:, 1]

    time_length_workload = arrs.sum()/1000/3600
    rng_ts = np.random.default_rng(seed=config.seed)
    st_s = rng_ts.random(num_models) * time_length_workload

    env_s = []

    for start_time in st_s:

        time_arr = np.cumsum(arrs) / 1000 / 3600
        assert start_time >= 0
        assert start_time < time_arr[-1]
        i_br = np.searchsorted(time_arr, start_time, side='right')
        arrs = np.r_[arrs[i_br:], arrs[:i_br]]
        sizes = np.r_[sizes[i_br:], sizes[:i_br]]

        job_gen = PyJobGenFile([sizes], [arrs], 0, 1.0e8 * 3600 * 1000, config.num_servers, config.seed, 1)

        env_s.append(pyenv.PyLoadBalanceEnv(config.num_servers, config.time_window * 1000, True, False, False, True,
                                            output_folder + 'data.log',
                                            config.lb_timeout_levels, [0.75, 0.85], config.max_num_retries,
                                            True, config.tw_safe_queue_size, config.tw_exit_queue_size, job_gen,
                                            job_gen.get_ptr(), config.seed, skip_log))
    return env_s
