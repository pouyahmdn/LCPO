import os
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


def transform_label(label: str):
    pieces = label.split('_')
    pieces[0] = pieces[0].upper()
    pieces = [piece for piece in pieces if piece != f't{trl[0]}' and piece not in ['mem', 'limit', 'no']]
    return '_'.join(pieces)


########################################################################################################################
# ##################################################### Load results ###################################################
########################################################################################################################

os.makedirs('./figures/', exist_ok=True)

trl = np.arange(4)
list_files = []
list_labels = []
tags = ['a2c_oracle_eval', 'a2c_continual', 'trpo_oracle_eval', 'trpo_continual', 'dqn_oracle_eval', 'dqn_continual',
        'sac_oracle_eval', 'sac_continual', 'lcpo_continual', 'lcpo_d_continual', 'lcpo_p_continual', 'mbcd_continual',
        'mbpo_continual', 'imbpo_continual']
for tag in tags:
    if 'oracle' in tag:
        list_files.extend([[f'./tests/{tag}_s{sd}_t{tind}/' for sd in range(10)] for tind in trl])
    else:
        list_files.extend([[f'./tests/{tag}_seed{sd}_trace_ind{tind}/' for sd in range(10)] for tind in trl])
    list_labels.extend([f'{tag}_t{tind}_para' for tind in trl])

assert len(list_labels) == len(list_files)

df_list = [[] for i in range(len(list_files))]
for ind in trange(len(list_files)):
    for path in list_files[ind]:
        variant = pd.read_pickle(f'{path}/tb.pkl')
        extracted = variant['State/episode_reward']
        ind_nan = ~np.isnan(extracted)
        extracted = extracted[ind_nan]
        assert len(extracted) > 0
        if 'trpo' in list_labels[ind] and 'continual' in list_labels[ind]:
            rate = 16
        else:
            rate = 1
        df_dict = {'reward': extracted, 'time': np.arange(len(extracted)) * rate, 'seed': len(df_list[ind])}
        df_list[ind].append(pd.DataFrame(df_dict))

for i in range(len(df_list)):
    if 'para' in list_labels[i]:
        rew_list = [df['reward'].mean() for df in df_list[i]]
        df_list[i] = df_list[i][np.argmax(rew_list)]
    else:
        df_list[i] = pd.concat(df_list[i], ignore_index=True)

mean_list = np.array([df[df['time'] > 30000]['reward'].mean() for df in df_list]).reshape((14, 4))
std_list = np.array([df[df['time'] > 30000].groupby(by='seed').mean()['reward'].std() / np.sqrt(
    df['seed'].nunique() - 1) * t(0.05, df['seed'].nunique() - 1) for df in df_list]).reshape((14, 4))

########################################################################################################################
# ################################################# Plot traces ########################################################
########################################################################################################################

fig, axes = plt.subplots(2, 4, figsize=(6.75, 2), sharey=True)
xticks = [[0, 5, 10, 15, 20]] * 2 + [[0, 2, 4, 6, 8]] * 2

for ind, i in enumerate(trl):
    samples = np.load(f'./traces/ou_tr{i}.npy')
    len_tot = len(samples) // 2
    wind_1, wind_2 = samples[:len_tot], samples[len_tot:]

    row = 0
    col = ind
    ax = axes[row][col]
    ax.plot(np.arange(0, len_tot, 200) / int(1e6), wind_1.reshape((-1, 200)).mean(axis=-1), color='C0')
    ax.set_yticks([-4, 0, 4])
    if ind == 0:
        ax.set_ylabel('X-axis')
    ax.set_xticks(xticks[ind])
    ax.set_xticklabels([''] * len(ax.get_xticks()))

    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    ax.set_title(f'Input Trace {ind + 1}')

    ax = axes[row + 1][col]
    ax.plot(np.arange(0, len_tot, 200) / int(1e6), wind_2.reshape((-1, 200)).mean(axis=-1), color='C2')
    ax.set_yticks([-4, 0, 4])
    if ind == 0:
        ax.set_ylabel('Y-axis')
    ax.set_xticks(xticks[ind])
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

fig.supxlabel('Samples ($\\times 1$ Million)')
fig.supylabel('Applied force (N)')
fig.tight_layout()
plt.subplots_adjust(bottom=0.2)

plt.savefig('figures/windy_pend_traces_single_horizontal.pdf', format='pdf')

########################################################################################################################
# ################################################# Plot time-series ###################################################
########################################################################################################################

fig, axes = plt.subplots(2, 2, figsize=(6.75, 2.25))
best_paras = np.argmax(mean_list[[0, 2, 4, 6], :], axis=0)
best_single = np.argmax(mean_list[[1, 3, 5, 7, 11, 12, 13], :], axis=0)
bases = ['A2C', 'TRPO', 'DDQN', 'SAC', 'MBCD', 'MBPO', 'IMBPO']
start_time = 30000

for i in range(len(trl)):
    row = i % 2
    col = i // 2
    ax = axes[row][col]

    for dfi, label, color, lst in zip(
            [i + 8 * len(trl), i + (best_single[i] * 2 + 1) * len(trl), i + best_paras[i] * 2 * len(trl)],
            ['LCPO', f'Best Continual ({bases[best_single[i]]})', f'Best Oracle ({bases[best_paras[i]]})'],
            ['C2', 'C3', 'C9'], ['-', '--', ':']):
        df = df_list[dfi]
        st = (df['time'].max() - start_time) // 100
        df['tint'] = (df['time'] / st).astype(int)
        df = df[df['time'] > start_time]
        rew = df.groupby('tint').mean()['reward']
        rew_ci = df.groupby(['tint', 'seed']).mean().groupby('tint').std()['reward'] / np.sqrt(
            df['seed'].nunique() - 1) * t(0.05, df['seed'].nunique() - 1)
        tint = df.groupby('tint').mean().reset_index()['tint'] * st

        ax.plot(tint * 200 / 1e6, rew, label=label, color=color, linestyle=lst)
        if 'Offline' not in label:
            ax.fill_between(tint * 200 / 1e6, rew + rew_ci, rew - rew_ci, color=color, alpha=0.1, linestyle=lst)
        if row == 0:
            # The line below is to avoid a warning
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(['' for k in ax.get_xticks()])

        ax.grid()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
        ax.set_title(f'Input Trace {i + 1}')
        ax.legend()

fig.supxlabel('Samples ($\\times 1$ Million)')
fig.supylabel('Episodic Reward')
fig.tight_layout()
plt.subplots_adjust(bottom=0.18, hspace=0.6, wspace=0.16, left=0.1)

plt.savefig('figures/windy_pend_time_series.pdf', format='pdf')

########################################################################################################################
# ################################################# Print table 1 ######################################################
########################################################################################################################

inder = [8, 11, 12, 13, 1, 3, 5, 7, 0, 2, 4, 6]
oracs = [0, 2, 4, 6]

label_row = f''
for k in inder:
    label_row += f' & {transform_label(list_labels[k * len(trl)])}'
print(label_row + ' \\\\')

for i in range(mean_list.shape[-1]):
    bin_bf = np.zeros(len(inder), dtype=bool)
    data_row = np.array([mean_list[k, i] for k in inder])
    bin_bf[np.argmax(data_row[:-4])] = 1
    bin_bf[-4 + np.argmax(data_row[-4:])] = 1
    print('\\midrule')
    str_row = f'\\multirow{{2}}{{*}}{{Input Trace {i + 1}}}'
    for ik, k in enumerate(inder):
        if bin_bf[ik]:
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
        if bin_bf[ik]:
            new_add = f'\\textbf{{+-{std_list[k, i]:.0f}}}'
        else:
            new_add = f'+-{std_list[k, i]:.0f}'
        if k == oracs[0]:
            new_add = f' & {new_add}'
        str_row += f' & {new_add}'
    str_row = str_row.replace('+-nan', '')
    str_row = str_row.replace('\\textbf{}', '')
    print(str_row + ' \\\\')

########################################################################################################################
# ################################################# Print table 2 ######################################################
########################################################################################################################

lcpo_ind = [8, 9, 10]
bases = [1, 3, 5, 7, 11, 12, 13]
oracs = [0, 2, 4, 6]

label_row = f''
for k in lcpo_ind:
    label_row += f' & {transform_label(list_labels[k * len(trl)])}'
label_row += f' & Best Continual & Best Pre-trained Oracle'
print(label_row + ' \\\\')

for i in range(mean_list.shape[-1]):
    data_row = np.array([mean_list[k, i] for k in bases])
    base_ind = bases[np.argmax(data_row)]
    data_row = np.array([mean_list[k, i] for k in oracs])
    orac_ind = oracs[np.argmax(data_row)]
    inder = lcpo_ind + [base_ind, orac_ind]
    data_row = np.array([mean_list[k, i] for k in inder])
    bin_bf = np.zeros(len(inder), dtype=bool)
    bin_bf[np.argmax(data_row[:-1])] = 1
    print('\\midrule')
    str_row = f'\\multirow{{2}}{{*}}{{Input Trace {i + 1}}}'
    for ik, k in enumerate(inder):
        if bin_bf[ik]:
            new_add = f'\\textbf{{{mean_list[k, i]:.0f}}}'
        else:
            new_add = f'{mean_list[k, i]:.0f}'
        if k in oracs:
            new_add = f'\\multirow{{2}}{{*}}{{{new_add}}}'
        str_row += f' & {new_add}'
    print(str_row + ' \\\\')
    str_row = ''
    for ik, k in enumerate(inder):
        if bin_bf[ik]:
            new_add = f'\\textbf{{+-{std_list[k, i]:.0f}}}'
        else:
            new_add = f'+-{std_list[k, i]:.0f}'
        str_row += f' & {new_add}'
    str_row = str_row.replace('+-nan', '')
    str_row = str_row.replace('\\textbf{}', '')
    print(str_row + ' \\\\')
