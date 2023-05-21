import os
import pickle
import time
from typing import List, Tuple, Dict
import numpy as np
from termcolor import colored
import struct
from functools import partial

from param import config


def compact(path: str):
    t_start = time.time()
    assert os.path.exists(os.path.join(path, 'models', f'model_{config.num_epochs}')) or config.saved_model, \
        "Final model not present"
    all_path = os.path.join(path, 'data.log_all')
    agent_path = os.path.join(path, 'data.log_agent')
    pickle_path = os.path.join(path, 'data.log_pkl')
    assert os.path.exists(all_path) and os.path.exists(agent_path)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            pkl = pickle.load(handle)
            if all(key in pkl for key in get_extracted_names()):
                print('Already there')
                return

    # All stats
    struct_fmt = '67fLB7x'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    with open(all_path, "rb") as f:
        data_all_raw = [struct_unpack(chunk) for chunk in iter(partial(f.read, struct_len), b'')]

    # Agent stats
    struct_fmt = '14f10I2fLB7x'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    with open(agent_path, "rb") as f:
        data_agent_raw = [struct_unpack(chunk) for chunk in iter(partial(f.read, struct_len), b'')]

    # Data Dict
    dict_save = extract_from_raw_data(data_all_raw, data_agent_raw)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Transforming to pickle took %.1f seconds' % (time.time()-t_start))


def extract_from_raw_data(all_raw: List[Tuple], agent_raw: List[Tuple]) -> Dict[str, np.ndarray]:
    all_np = np.array(all_raw)
    assert all_np.ndim == 2
    assert all_np.shape[1] == 69
    assert np.all(all_np[1:, 1] >= all_np[:-1, 2])
    return_arrs = [all_np[:, i] for i in range(all_np.shape[1])]
    agent_np = np.array(agent_raw)
    assert agent_np.ndim == 2
    assert agent_np.shape[1] == 28
    assert np.all(agent_np[1:, 1] >= agent_np[:-1, 2])
    return_arrs.extend([agent_np[:, i] for i in range(14)])
    return_arrs.append(agent_np[:, 14:14+len(config.lb_timeout_levels)])
    return_arrs.extend([agent_np[:, i] for i in range(24, 28)])

    if len(return_arrs) != len(get_extracted_names()):
        print(colored(f'WARNING: Number of extractions ({len(return_arrs)}) do not match '
                      f'labels ({len(get_extracted_names())})', 'red'))
    dict_save = {key: data for key, data in zip(get_extracted_names(), return_arrs)}
    return dict_save


def get_extracted_names() -> List[str]:
    stats = ['avg', 'min', 'max', 'med', '95', '97', '99']
    all_datas = ['time', 'delay', 'first_duration', 'duration', 'size', 'slow_down_first', 'slow_down',
                 'slow_down_size', 'qdelay']
    agent_datas = ['time_act', 'action']
    names_all = [f'{data}_{stat}' for data in all_datas for stat in stats]
    names_all_extra = ['inflation_avg', 'model_index_avg', 'tw_avg', 'time_len', 'len', 'trace']
    names_agent = [f'{data}_{stat}' for data in agent_datas for stat in stats]
    names_agent_extra = ['histogram', 'entropy', 'time_act_len', 'act_len', 'act_trace']
    return names_all + names_all_extra + names_agent + names_agent_extra
