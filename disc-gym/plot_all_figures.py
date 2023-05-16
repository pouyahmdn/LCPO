import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
import scipy.stats

plt.rcParams.update({
    "text.usetex": True,
    'legend.fontsize': 6,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.labelsize': 8,
    'figure.titlesize': 8,
    'hatch.linewidth': 0.5
})


def t(alpha, gl):
    return scipy.stats.t.ppf(1 - (alpha / 2), gl)


########################################################################################################################
# ##################################################### Load results ###################################################
########################################################################################################################


list_files = [[f'./tests/a2c_pendulum_disc_seed{sd}/'] for sd in range(10)]
list_files.extend([[f'./tests/trpo_pendulum_disc_seed{sd}/'] for sd in range(10)])
list_files.extend([[f'./tests/dqn_pendulum_disc_seed{sd}/'] for sd in range(10)])
list_files.extend([[f'./tests/sac_pendulum_disc_seed{sd}/'] for sd in range(10)])
list_files.extend([[f'./tests/mbpo_pendulum_disc_seed{sd}/'] for sd in range(10)])
list_labels = []
list_labels.extend([f'a2c_s{sd}' for sd in range(10)])
list_labels.extend([f'trpo_s{sd}' for sd in range(10)])
list_labels.extend([f'dqn_s{sd}' for sd in range(10)])
list_labels.extend([f'sac_s{sd}' for sd in range(10)])
list_labels.extend([f'mbpo_s{sd}' for sd in range(10)])

assert len(list_labels) == len(list_files)

df_list = [[] for i in range(len(list_files))]
for ind in trange(len(list_files)):
    for path in list_files[ind]:
        variant = pd.read_pickle(f'{path}/tb.pkl')
        extracted = variant['State/episode_reward']
        extracted = extracted[~np.isnan(extracted)]
        assert len(extracted) > 0
        if 'trpo' in list_labels[ind]:
            rate = 16
        else:
            rate = 1
        df_list[ind].append(pd.DataFrame({'reward': extracted, 'time': np.arange(len(extracted)) * rate, 'seed': int(list_labels[ind][-1])}))
for i in range(len(df_list)):
    df_list[i] = pd.concat(df_list[i], ignore_index=True)
mean_list = np.array([df[(df['time'] > 8000) & (df['time'] < 9000)]['reward'].mean() for df in df_list]).reshape((5, 10))
std_list = mean_list.std(axis=-1)/np.sqrt(mean_list.shape[1]-1)*t(0.05, mean_list.shape[1]-1)

inder = [0, 1, 2, 3, 4]
print(' & '.join([agent[:-3].upper() for agent in list_labels[::10]]), '\\\\')
print('\\midrule')
str_row = f'Episodic Reward'
for k in inder:
    str_row += f' & {mean_list[k, :].mean():.0f}+-{std_list[k]:.0f} '
print(str_row + '\\\\')
