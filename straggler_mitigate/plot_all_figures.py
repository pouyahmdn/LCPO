import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
import scipy.stats
import time
import pickle
from numba import jit

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


def load_exp_list(list_files, agent_alias, print_items=True):
    e_list = []
    for i in trange(len(list_files)):
        t1 = time.time()
        with open(list_files[i] + '/data.log_pkl', 'rb') as handle:
            pkl_rick = pickle.load(handle)
        for key in pkl_rick:
            pkl_rick[key] = np.array(pkl_rick[key])
        e_list.append(pkl_rick)
        if print_items:
            print(f'{agent_alias[i]} loaded at index {i}!!')
        e_list[-1]['alias'] = agent_alias[i]
    return e_list


@jit
def ewma(data: np.ndarray, alpha: float) -> np.ndarray:
    smoothed = np.zeros_like(data)
    weight = 0
    smoothed[0] = data[0]
    summer = 0
    for k in range(data.shape[0]):
        weight = 1 * (1 - alpha) + alpha * weight
        summer = data[k] * (1 - alpha) + alpha * summer
        smoothed[k] = summer / weight
    return smoothed


@jit
def sum_block(tbuck: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    arr_size = np.zeros(np.max(tbuck)+1, dtype=float)
    for i in range(len(sizes)):
        arr_size[tbuck[i]] += sizes[i]
    return arr_size


def t(alpha, gl):
    return scipy.stats.t.ppf(1-(alpha/2), gl)


def transform_label(label: str):
    pieces = label.split('_')
    pieces[0] = pieces[0].upper()
    pieces = [piece for piece in pieces if piece != f't{trl[0]}' and piece not in ['mem', 'limit', 'no']]
    return '_'.join(pieces)


########################################################################################################################
# ##################################################### Plot traces ####################################################
########################################################################################################################

os.makedirs('./figures/', exist_ok=True)

traces = ['./traces/real_tr0.npy', './traces/real_tr1.npy']
comptraces = [[] for _ in range(len(traces))]
window = 0.5

for ind, path in enumerate(traces):
    workload = np.load(path)
    sizes = workload[:, 0]
    ints = workload[:, 1]
    tints = np.cumsum(ints) / 1000
    tbuck = (tints / window).astype(int)
    assert tbuck.shape == sizes.shape
    arr_rate = np.bincount(tbuck)
    arr_size = sum_block(tbuck, sizes)
    arr_size /= (arr_rate + 1e-8)
    assert arr_size.shape == arr_rate.shape
    comptraces[ind] = [arr_rate / window, arr_size, np.arange(len(arr_rate)) * window / 3600]

fig, axes = plt.subplots(2, 2, figsize=(3.25, 2), sharex=True)

for ind, (arr_rate, arr_size, arr_time) in enumerate(comptraces):
    row = 0
    col = ind
    ax = axes[row][col]
    ax.plot(arr_time, ewma(arr_rate, 0.999), color='C0')
    #     ax.set_yticks([-4, 0, 4])
    if ind == 0:
        ax.set_ylabel('Arrival Rate')

    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    ax.set_title(f'Context Trace {ind + 1}')

    ax = axes[row + 1][col]
    ax.plot(arr_time, ewma(arr_size, 0.999), color='C2')
    #     ax.set_yticks([-4, 0, 4])
    if ind == 0:
        ax.set_ylabel('Processing Time')
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    ax.set_ylim(bottom=-5 / 100 * ax.get_ylim()[1])

fig.supxlabel('Time (hrs)')
fig.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig('figures/lb_traces.pdf', format='pdf')


########################################################################################################################
# ##################################################### Load results ###################################################
########################################################################################################################


trl = np.arange(2)
list_files = []
list_labels = []
tags = ['eval_a2c_oracle', 'a2c_continual',
        'eval_trpo_oracle', 'trpo_continual',
        'eval_dqn_oracle', 'dqn_continual',
        'eval_sac_oracle', 'sac_continual',

        'lcpo_continual_ablation_trace_ind%d_seed%d_lcpo_thresh-5',
        'lcpo_continual_ablation_trace_ind%d_seed%d_lcpo_thresh-6',
        'lcpo_continual_ablation_trace_ind%d_seed%d_lcpo_thresh-7',

        'mbcd_continual', 'mbpo_continual', 'ewcpp_continual']
for tag in tags:
    if tag.startswith('lcpo_continual_ablation'):
        list_files.extend([f'./tests/{tag % (tind, sd)}/' for tind in trl for sd in range(10)])
    else:
        list_files.extend([f'./tests/{tag}_trace_ind{tind}_seed{sd}/' for tind in trl for sd in range(10)])
    list_labels.extend([f'{tag}_t{tind}_para' for tind in trl])

e_list = load_exp_list(list_files, [f'{i}' for i in range(len(list_files))], print_items=False)

labels = [
    'Oracle A2C',
    'A2C',
    'Oracle TRPO',
    'TRPO',
    'Oracle DQN',
    'DQN',
    'Oracle SAC',
    'SAC',

    'LCPO -5',
    'LCPO -6',
    'LCPO -7',

    'MBCD',
    'MBPO',
    'Online EWC'
]
labels = sum([[label + ' - T0', label + ' - T1'] for label in labels], [])

df_list = [[] for i in range(len(labels))]
for variant in range(len(list_files)):
    extracted = e_list[variant]['delay_95']
    time_arr = e_list[variant]['time_max'] / 3600 / 1000
    index_start = np.searchsorted(time_arr, 27000 * 128 * 0.5 / 3600)
    index_end = np.searchsorted(time_arr, 162000 * 128 * 0.5 / 3600)
    assert index_start < len(extracted)
    ind = variant // 10
    df_list[ind].append(pd.DataFrame(
        {'delay': extracted[index_start:index_end], 'time': time_arr[index_start:index_end], 'seed': variant % 10}))
for i in range(len(df_list)):
    if 'Parallel' in labels[i]:
        rew_list = [df['delay'].mean() for df in df_list[i]]
        df_list[i] = df_list[i][np.argmin(rew_list)]
    else:
        df_list[i] = pd.concat(df_list[i], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.5))
mean_list = np.array([df['delay'].mean() for df in df_list]).reshape((13, 2))
std_list = np.array(
    [df.groupby(by='seed').mean()['delay'].std() / np.sqrt(df['seed'].nunique() - 1) * t(0.05, df['seed'].nunique() - 1)
     for df in df_list]).reshape((13, 2))
best_paras = np.argmin(mean_list[[0, 2, 4, 6], :], axis=0)
best_single = np.argmin(mean_list[[1, 3, 5, 7, 11, 12, 13], :], axis=0)
bases = ['A2C', 'TRPO', 'DDQN', 'SAC', 'MBCD', 'MBPO', 'Online EWC']

for i in range(mean_list.shape[-1]):
    row = i % 2
    ax = axes[row]

    for dfi, label, color, lst in zip([i + 10 * mean_list.shape[-1], i + (best_single[i] * 2 + 1) * mean_list.shape[-1],
                                       i + best_paras[i] * 2 * mean_list.shape[-1]],
                                      ['LCPO Cons', f'Best Baseline ({bases[best_single[i]]})',
                                       f'Prescient ({bases[best_paras[i]]})'],
                                      ['C2', 'C3', 'C9'], ['-', '--', ':']):
        df = df_list[dfi]
        st = (df['time'].max() - df['time'].min()) // 100
        df['tint'] = (df['time'] / st).astype(int)
        rew = df.groupby('tint').mean()['delay']
        rew_ci = df.groupby(['tint', 'seed']).mean().groupby('tint').std()['delay'] / np.sqrt(
            df['seed'].nunique() - 1) * t(0.05, df['seed'].nunique() - 1)
        tint = df.groupby('tint').mean().reset_index()['tint'] * st

        ax.plot(tint, rew, label=label, color=color, linestyle=lst)
        if 'Oracle' not in label:
            ax.fill_between(tint, rew + rew_ci, rew - rew_ci, color=color, alpha=0.1, linestyle=lst)

        ax.grid()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
        ax.set_title(f'Context Trace {i + 1}')
        ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.03, 1.35, 1.06, 0.2), borderaxespad=0, mode="expand")

fig.supxlabel('Time (hrs)')
fig.supylabel('Tail latency (ms)')
fig.tight_layout()
plt.subplots_adjust(bottom=0.25, top=0.7, hspace=0.6, wspace=0.16, left=0.1)

plt.savefig('figures/lb_time_series.pdf', format='pdf')

################################################################################################################################

fig, ax = plt.subplots(1, 1, figsize=(6.5, 1.4))
mean_list = np.array([df['delay'].mean() for df in df_list]).reshape((13, 2))
std_list = np.array(
    [df.groupby(by='seed').mean()['delay'].std() / np.sqrt(df['seed'].nunique() - 1) * t(0.05, df['seed'].nunique() - 1)
     for df in df_list]).reshape((13, 2))
best_paras = np.argmin(mean_list[[0, 2, 4, 6], :], axis=0)
best_single = np.argmin(mean_list[[1, 3, 5, 7, 11, 12, 13], :], axis=0)
bases = ['A2C', 'TRPO', 'DDQN', 'SAC', 'MBCD', 'MBPO', 'Online EWC']

i = 0
for dfi, label, color, lst in zip([i + 10 * mean_list.shape[-1], i + (best_single[i] * 2 + 1) * mean_list.shape[-1],
                                   i + best_paras[i] * 2 * mean_list.shape[-1]],
                                  ['LCPO Cons', f'Best Baseline ({bases[best_single[i]]})',
                                   f'Prescient ({bases[best_paras[i]]})'],
                                  ['C2', 'C3', 'C9'], ['-', '--', ':']):
    df = df_list[dfi]
    st = (df['time'].max() - df['time'].min()) // 100
    df['tint'] = (df['time'] / st).astype(int)
    rew = df.groupby('tint').mean()['delay']
    rew_ci = df.groupby(['tint', 'seed']).mean().groupby('tint').std()['delay'] / np.sqrt(df['seed'].nunique() - 1) * t(
        0.05, df['seed'].nunique() - 1)
    tint = df.groupby('tint').mean().reset_index()['tint'] * st

    ax.plot(tint, rew, label=label, color=color, linestyle=lst)
    if 'Oracle' not in label:
        ax.fill_between(tint, rew + rew_ci, rew - rew_ci, color=color, alpha=0.1, linestyle=lst)

    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    ax.set_title(f'Context Trace {i + 1}')
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0.25, 1.35, 0.5, 0.2), borderaxespad=0, mode="expand")

fig.supxlabel('Time (hrs)')
fig.supylabel('Tail latency (ms)')
fig.tight_layout()
plt.subplots_adjust(bottom=0.25, top=0.7, hspace=0.6, wspace=0.16, left=0.1)

plt.savefig('figures/lb_time_series_only_i1.pdf', format='pdf')

########################################################################################################################
# ################################################# Print table 2 ######################################################
########################################################################################################################
inder = [8, 9, 10, 11, 12, 13, 1, 3, 5, 7, 0, 2, 4, 6]
oracs = [0, 2, 4, 6]

label_row = f''
for k in inder:
    label_row += f' & {labels[k*2][:-5]}'
print(label_row + ' \\\\')

for i in range(mean_list.shape[-1]):
    bin_bf = np.zeros(len(inder), dtype=bool)
    data_row = np.array([mean_list[k, i] for k in inder])
    bin_bf[np.argmin(data_row[:-4])] = 1
    bin_bf[-4+np.argmin(data_row[-4:])] = 1
    print('\\midrule')
    str_row = f'\\multirow{{2}}{{*}}{{Context Trace {i+1}}}'
    for ik, k in enumerate(inder):
        if bin_bf[ik] == True:
            new_add = f'\\textbf{{{mean_list[k, i]:.0f}}}'
        else:
            new_add = f'{mean_list[k, i]:.0f}'
        if k in oracs:
            new_add = f'\\multirow{{2}}{{*}}{{{new_add}}}'
        if k == oracs[0]:
            new_add = f' & {new_add}'
        str_row += f' & {new_add}'
    print(str_row + ' \\\\')
    str_row = ''
    for ik, k in enumerate(inder):
        if bin_bf[ik] == True:
            new_add = f'\\textbf{{+-{std_list[k, i]:.0f}}}'
        else:
            new_add = f'+-{std_list[k, i]:.0f}'
        if k == oracs[0]:
            new_add = f' & {new_add}'
        str_row += f' & {new_add}'
    str_row = str_row.replace('+-nan', '')
    str_row = str_row.replace('\\textbf{}', '')
    print(str_row + ' \\\\')
