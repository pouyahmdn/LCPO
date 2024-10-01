import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
import scipy.stats
from typing import List
from matplotlib.patches import Patch

# We use a latex engine for texts in matplotlib. If you do not have a latex engine on your machine, set "text.usetex" to
# False.
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


def load_data(data_files: List[List[str]], data_labels: List[str]) -> List[pd.DataFrame]:
    assert len(data_files) == len(data_labels)

    data_df = [[] for _ in range(len(data_files))]
    for i in trange(len(data_files)):
        for data_path in data_files[i]:
            data_variant = pd.read_pickle(f'{data_path}/tb.pkl')
            data_extracted = data_variant['State/episode_reward']
            elapsed_extracted = data_variant['Time/elapsed']
            data_ind_nan = ~np.isnan(data_extracted)
            data_extracted = data_extracted[data_ind_nan]
            elapsed_extracted = elapsed_extracted[data_ind_nan]
            assert len(data_extracted) > 0
            assert len(elapsed_extracted) > 0
            if 'trpo' in data_labels[i] and 'continual' in data_labels[i]:
                data_rate = 16
            else:
                data_rate = 1
            data_df_dict = {'reward': data_extracted,
                            'elapsed': elapsed_extracted,
                            'time': np.arange(len(data_extracted)) * data_rate,
                            'seed': len(data_files[i])}
            data_df[i].append(pd.DataFrame(data_df_dict))

    consolidated_data_df = []
    for i in range(len(data_df)):
        if 'para' in data_labels[i]:
            data_rew_list = [df['reward'].mean() for df in data_df[i]]
            consolidated_data_df.append(data_df[i][np.argmax(data_rew_list)])
        else:
            consolidated_data_df.append(pd.concat(data_df[i], ignore_index=True))

    return consolidated_data_df


def t(alpha: float, gl: int):
    return scipy.stats.t.ppf(1 - (alpha / 2), gl - 1) / np.sqrt(gl - 1)


def get_appropriate_fmt(float_num: float or np.ndarray) -> str:
    if np.abs(float_num) >= 100:
        num_fmt = '.0f'
    elif np.abs(float_num) >= 10:
        num_fmt = '.1f'
    else:
        num_fmt = '.2f'
    return num_fmt


########################################################################################################################
# ##################################################### Load results ###################################################
########################################################################################################################

os.makedirs('./figures/', exist_ok=True)

num_seeds = 25
num_seeds_oracle = 5
num_seeds_ablate = 5
num_seeds_mbpo_mbcd = 5
trl = np.arange(4)
list_files = []
list_labels = []
tags = [
    'a2c_oracle_eval', 'a2c_continual',
    'trpo_oracle_eval', 'trpo_continual',
    'dqn_oracle_eval', 'dqn_continual',
    'sac_oracle_eval', 'sac_continual',

    'lcpo_continual',

    'lcpo_continual_buffer_size_ablation_seed{sd}_ood_subsample1',
    'lcpo_continual_buffer_size_ablation_seed{sd}_ood_subsample100',
    'lcpo_continual_buffer_size_ablation_seed{sd}_ood_subsample40000',
    'lcpo_continual_buffer_size_ablation_seed{sd}_ood_subsample800000',

    'lcpo_continual_thres_ablation_seed{sd}_lcpo_thresh0.25',
    'lcpo_continual_thres_ablation_seed{sd}_lcpo_thresh0.5',
    'lcpo_continual_thres_ablation_seed{sd}_lcpo_thresh1',
    'lcpo_continual_thres_ablation_seed{sd}_lcpo_thresh2',
    'lcpo_continual_thres_ablation_seed{sd}_lcpo_thresh4',

    'mbcd_continual', 'mbpo_continual', 'ewcpp_continual', 'slide_ogd_continual', 'bfdqn_continual', 'clear_continual',

    'lcpo_continual_mahala_ablation_seed{sd}_lcpo_thresh-3', 'lcpo_continual_mahala_ablation_seed{sd}_lcpo_thresh-6',]
env_s = ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
for env in env_s:
    for tag in tags:
        if 'oracle' in tag:
            list_files.extend([[f'./tests/{tag}_s{sd}_env{env}_t{tind}/' for sd in range(num_seeds_oracle)] for tind in trl])
        elif 'seed{sd}' in tag:
            list_files.extend([[f'./tests/{tag.format(sd=sd)}_env{env}_trace_ind{tind}/' for sd in range(num_seeds_ablate)]
                               for tind in trl])
        else:
            if 'mbpo' in tag or 'mbcd' in tag:
                seed_count = num_seeds_mbpo_mbcd
            else:
                seed_count = num_seeds
            list_files.extend([[f'./tests/{tag}_seed{sd}_env{env}_trace_ind{tind}/' for sd in range(seed_count)]
                               for tind in trl])
        list_labels.extend([f'{tag}_{env}_t{tind}' for tind in trl])

df_list = load_data(list_files, list_labels)

mean_list = np.array([df[df['time'] > 30000]['reward'].mean() for df in df_list]).reshape(
    (len(env_s), len(tags), len(trl)))
std_list = np.array(
    [df[df['time'] > 30000].groupby(by='seed').mean()['reward'].std() * t(0.05, df['seed'].nunique()) for df in
     df_list]).reshape(
    (len(env_s), len(tags), len(trl)))

mean_list_runtime = np.array([df[df['time'] > 30000]['elapsed'].mean() for df in df_list]).reshape((len(env_s), len(tags), len(trl)))
std_list_runtime = np.array([df[df['time'] > 30000].groupby(by='seed').mean()['elapsed'].std()/np.sqrt(df['seed'].nunique()-1)*t(0.05, df['seed'].nunique()-1) for df in df_list]).reshape((len(env_s), len(tags), len(trl)))

# ##################################################### Traces #########################################################
# ################################################### Figure 6 #########################################################

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
    ax.set_title(f'Context Trace {ind + 1}')

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


# ##################################################### Traces #########################################################
# ################################################### Figure 2 #########################################################


def cpd_arr_direct(wind_1d: np.ndarray, ns: float, w_thres: float) -> np.ndarray:
    task_i = np.zeros_like(wind_1d, dtype=int)
    last_i = 0
    x_1 = wind_1d[0]
    x_2 = wind_1d[0] ** 2
    s_score = 0
    for wind_i in trange(1, len_tot - 1):
        mu = x_1 / (wind_i - last_i)
        var = x_2 / (wind_i - last_i) - mu ** 2
        if wind_i - last_i > 5000:
            s_score = max(0, s_score + (wind_1d[wind_i + 1] - mu) ** 2 / var - ns ** 2)
            if s_score > w_thres:
                last_i = wind_i
                s_score = 0
                x_1 = 0
                x_2 = 0
                task_i[wind_i] = 1
        x_1 += wind_1d[wind_i]
        x_2 += wind_1d[wind_i] ** 2
    return task_i


samples = np.load(f'./traces/ou_tr1.npy')
len_tot = len(samples) // 2
wind = samples[len_tot:]
inferred_task_change_big = cpd_arr_direct(wind, 3.1, 1000)
inferred_task_change_sml = cpd_arr_direct(wind, 3.0, 1000)

fig, ax = plt.subplots(1, 1, figsize=(3.25, 1.25))

ax.plot(np.arange(0, len_tot, 200) / int(1e6), wind.reshape((-1, 200)).mean(axis=-1), color='C2')
for i in np.where(inferred_task_change_big == 1)[0]:
    ax.axvline(i / int(1e6), color='C3')
ax.set_yticks([-4, 0, 4])
ax.set_ylabel('$z_t$')
ax.set_xlabel('$t$')
ax.set_title('$\\sigma_{mbcd}=3.1$')
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.set_xticklabels(['0', '4M', '8M', '12M', '16M', '20M'])
ax.set_yticks([-4, -2, 0, 2])
ax.grid()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

fig.tight_layout()

plt.savefig('figures/ood_vs_cpd_30.pdf', format='pdf')

fig, ax = plt.subplots(1, 1, figsize=(3.25, 1.25))

for i in np.where(inferred_task_change_sml == 1)[0]:
    ax.axvline(i / int(1e6), color='C3', linewidth=0.5)
ax.plot(np.arange(0, len_tot, 200) / int(1e6), wind.reshape((-1, 200)).mean(axis=-1), color='C2')
ax.set_yticks([-4, 0, 4])
ax.set_ylabel('$z_t$')
ax.set_xlabel('$t$')
ax.set_title('$\\sigma_{mbcd}=3$')
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.set_xticklabels(['0', '4M', '8M', '12M', '16M', '20M'])
ax.set_yticks([-4, -2, 0, 2])
ax.grid()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

fig.tight_layout()

plt.savefig('figures/ood_vs_cpd_31.pdf', format='pdf')

# ##################################################### All CDFs #######################################################
# ################################################### Figure 3a ########################################################

fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
line_st = [
    ([0, 2, 4, 6], 'Best Prescient', 'C7', 'dashed'),
    ([20], 'Online EWC', 'C9', (0, (2, 1, 1, 1))),
    ([21], 'Sliding OGD', 'C6', (0, (2, 1, 1, 1))),
    ([23], 'CLEAR', 'C1', (0, (2, 1, 1, 1))),
    ([22], 'BFDQN', 'C4', (0, (3, 2, 2))),
    ([1], 'A2C', 'C0', 'dotted'),
    ([7], 'SAC', 'C5', (0, (3, 1, 1, 1, 1, 1))),
    ([3], 'TRPO', 'C1', 'dashed'),
    ([5], 'DDQN', 'C4', (2, (1, 1, 1))),
    ([19], 'MBPO', 'C8', (5, (10, 3))),
    ([18], 'MBCD', 'C3', (1, (3, 1, 1, 3))),
    ([8], 'LCPO', 'C2', 'solid'),
]

for o, lbl, col, styl in line_st:
    samp = []
    for j in range(len(env_s)):
        for k in range(len(trl)):
            best = mean_list[j, :, k].max()
            ag = mean_list[j, o, k].max()
            base = mean_list[j, :, k].min()
            assert base < best
            samp.append((ag - base) / (best - base))
    x = np.sort(samp)
    y = np.linspace(0, 100, len(x))
    ax.plot(x, y, label=lbl, color=col, linestyle=styl)

ax.grid()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(-0.19, 1.02, 1.27, 0.1), mode="expand", handlelength=3)

fig.supylabel('CDF (\\%)')
fig.supxlabel('Normalized Return')
fig.tight_layout()
plt.subplots_adjust(bottom=0.18, hspace=1.1, wspace=0.16, left=0.15, top=0.77)

plt.savefig('figures/windy_return_cdf.pdf', format='pdf')

# #################################################### All CDFs ########################################################
# #################################################### Table 5 #########################################################

env_names = ['Pendulum', 'Inverse Pendulum', 'Inverse Double Pendulum', 'Hopper', 'Reacher']
labels_bases = ['LCPO', 'Online EWC', 'Sliding OGD', 'CLEAR', 'BFDQN', 'MBCD', 'MBPO', 'A2C', 'TRPO', 'DDQN', 'SAC']
labels_oracs = ['A2C', 'TRPO', 'DDQN', 'SAC']

bases = [8, 20, 21, 23, 22, 18, 19, 1, 3, 5, 7]
oracs = [0, 2, 4, 6]

print('\\begin{tabular}{c l ' + ' '.join(['c'] * (len(bases) + len(oracs) + 1)) + '}')
print('\\toprule')
print(f'& & \\multicolumn{{{len(bases)}}}{{c}}{{\\textbf{{Online Learning}}}} '
      f'& & \\multicolumn{{{len(oracs)}}}{{c}}{{\\textbf{{Prescient Policies}}}} \\\\')
cmid = 3 + len(bases) - 1
cend = 3 + len(bases) + len(oracs)
print(f'\\cmidrule{{3-{cmid}}} \\cmidrule{{{cmid + 2}-{cend}}}')

label_row = f'&'
for k in labels_bases:
    label_row += f' & {k}'
label_row += " &"
for k in labels_oracs:
    label_row += f' & {k}'
print(label_row + ' \\\\')
print()
print('\\midrule')
print()

for i in range(mean_list.shape[0]):
    for j in range(mean_list.shape[-1]):
        inder = bases + oracs
        data_row = np.array([mean_list[i, k, j] for k in inder])
        bin_bf = np.zeros(len(inder), dtype=bool)
        bin_bf[np.argmax(data_row[:-len(oracs)])] = 1
        bin_bf[len(bases) + np.argmax(data_row[-len(oracs):])] = 1
        if j == 0:
            print('\n\\midrule\n')
            str_row = '\\multirow{10}{*}{\\rotatebox[origin=c]{90}{' + env_names[i] + '}} & '
        else:
            print(f'\\cmidrule{{2-{cend}}}')
            str_row = ' & '
        str_row += f'\\multirow{{2}}{{*}}{{Context Trace {j + 1}}}'
        for ik, k in enumerate(inder):
            num = mean_list[i, k, j]

            fmt = get_appropriate_fmt(num)

            if bin_bf[ik]:
                new_add = f'\\textbf{{{num:{fmt}}}}'
            else:
                new_add = f'{num:{fmt}}'

            if k == bases[-1]:
                new_add = f'{new_add} & '

            if k in oracs:
                new_add = f'\\multirow{{2}}{{*}}{{{new_add}}}'

            str_row += f' & {new_add}'
        print(str_row + ' \\\\')
        str_row = ' & '
        for ik, k in enumerate(inder):
            num = std_list[i, k, j]

            fmt = get_appropriate_fmt(num)

            if bin_bf[ik]:
                new_add = f'\\textbf{{$\\pm${num:{fmt}}}}'
            else:
                new_add = f'$\\pm${num:{fmt}}'

            if k == bases[-1]:
                new_add = f'{new_add} & '
            str_row += f' & {new_add}'
        str_row = str_row.replace('$\\pm$nan', '')
        str_row = str_row.replace('\\textbf{}', '')
        print(str_row + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')

# ################################################## Sigma Ablation ####################################################
# ################################################### Figure 3b ########################################################

fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
line_st = [
    ([13], 'LCPO $\sigma^2=0.25$', 'k', 1, (0, (1, 1))),
    ([14], 'LCPO $\sigma^2=0.5$', 'C3', 1, (0, (3, 1, 1, 1, 1, 1))),
    ([15], 'LCPO $\sigma^2=1.0$', 'C2', 1, 'solid'),
    ([16], 'LCPO $\sigma^2=2.0$', 'C5', 1, (5, (10, 1, 5, 1))),
    ([17], 'LCPO $\sigma^2=4.0$', 'C6', 1, (5, (2, 1, 4, 1))),
    ([1], 'A2C', 'C0', 1, 'dotted'),
    ([24], r'LCPO $\sigma_{\text{MHD}}^2=6.0$', 'C8', 1, (1, (3, 1, 1, 3))),
    ([25], r'LCPO $\sigma_{\text{MHD}}^2=12.0$', 'C9', 1, (0, (2, 1, 1, 1))),
    ([0, 2, 4, 6], 'Best Prescient', 'C7', 1, 'dashed'),
]

for o, lbl, col, alph, styl in line_st:
    samp = []
    for j in range(len(env_s)):
        for k in range(len(trl)):
            best = mean_list[j, :, k].max()
            ag = mean_list[j, o, k].max()
            base = mean_list[j, :, k].min()
            assert base < best
            samp.append((ag - base) / (best - base))
    x = np.sort(samp)
    y = np.linspace(0, 100, len(x))
    ax.plot(x, y, label=lbl, color=col, linestyle=styl, alpha=alph)

ax.grid()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
ax.set_xlim(left=0.35, right=1.05)
ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.19, 1.02, 1.27, 0.1), mode="expand", handlelength=3)

fig.supylabel('CDF (\\%)')
fig.supxlabel('Normalized Return')
fig.tight_layout()
plt.subplots_adjust(bottom=0.18, hspace=1.1, wspace=0.16, left=0.15, top=0.77)

plt.savefig('figures/windy_return_sigma.pdf', format='pdf')

# ################################################## Sigma Ablation ####################################################
# #################################################### Table 6 #########################################################

env_names = ['Pendulum', 'Inverse Pendulum', 'Inverse Double Pendulum', 'Hopper', 'Reacher']
labels_lcpo = ['$\\sigma^2=0.25$', '$\\sigma^2=0.5$', '$\\sigma^2=1$', '$\\sigma^2=2$', '$\\sigma^2=4$', r'$\sigma_{\text{MHD}}^2=6$', r'$\sigma_{\text{MHD}}^2=12$']
labels_bases = ['Online EWC', 'Sliding OGD', 'CLEAR', 'BFDQN', 'MBCD', 'MBPO', 'A2C', 'TRPO', 'DDQN', 'SAC']
labels_oracs = ['A2C', 'TRPO', 'DDQN', 'SAC']

lcpo_v1_ind = [13, 14, 15, 16, 17]
lcpo_v2_ind = [24, 25]
bases = [20, 21, 23, 22, 18, 19, 1, 3, 5, 7]
oracs = [0, 2, 4, 6]

print('\\begin{tabular}{c l ' + ' '.join(['c'] * (len(lcpo_v1_ind) + len(lcpo_v2_ind) + 3)) + '}')
print('\\toprule')
print(f'& & \\multicolumn{{{len(lcpo_v1_ind)}}}{{c}}{{\\textbf{{LCPO (L2)}}}} & & \\multicolumn{{{len(lcpo_v2_ind)}}}{{c}}{{\\textbf{{LCPO (MHD)}}}} & & \\\\')
cmid = 3 + len(lcpo_v1_ind) - 1
cmid_v2 = 3 + len(lcpo_v1_ind) + len(lcpo_v2_ind) - 1
cend = 3 + len(lcpo_v1_ind) + len(lcpo_v2_ind) + 2
print(f'\\cmidrule{{3-{cmid}}}')
print(f'\\cmidrule{{{cmid+2}-{cmid_v2}}}')

label_row = f'&'
for i, k in enumerate(labels_lcpo):
    label_row += f' & {k}'
    if i == len(lcpo_v1_ind) - 1:
        label_row += ' & '
label_row += " & Best Baseline & Best Prescient \\\\"
print(label_row)
print()
print('\\midrule')
print()

for i in range(mean_list.shape[0]):
    for j in range(mean_list.shape[-1]):
        data_row = np.array([mean_list[i, k, j] for k in bases])
        base_ind = bases[np.argmax(data_row)]
        base_label = labels_bases[np.argmax(data_row)]

        data_row = np.array([mean_list[i, k, j] for k in oracs])
        orac_ind = oracs[np.argmax(data_row)]
        orac_label = labels_oracs[np.argmax(data_row)]

        inder = lcpo_v1_ind + lcpo_v2_ind + [base_ind, orac_ind]
        data_row = np.array([mean_list[i, k, j] for k in inder])
        bin_bf = np.zeros(len(inder), dtype=bool)
        bin_bf[np.argmax(data_row[:-1])] = 1

        if j == 0:
            print('\n\\midrule\n')
            str_row = '\\multirow{10}{*}{\\rotatebox[origin=c]{90}{' + env_names[i] + '}} & '
        else:
            print(f'\\cmidrule{{2-{cend}}}')
            str_row = ' & '
        str_row += f'\\multirow{{2}}{{*}}{{Context Trace {j + 1}}}'
        for ik, k in enumerate(inder):
            num = mean_list[i, k, j]
            fmt = get_appropriate_fmt(num)
            if bin_bf[ik]:
                new_add = f'\\textbf{{{num:{fmt}}}}'
            else:
                new_add = f'{num:{fmt}}'
            if k in bases:
                new_add = f'{new_add} ({base_label})'
            if k == lcpo_v1_ind[-1]:
                new_add = f'{new_add} &'
            if k in oracs:
                new_add = f'\\multirow{{2}}{{*}}{{{new_add} ({orac_label})}}'
            str_row += f' & {new_add}'
        print(str_row + ' \\\\')
        str_row = ' & '
        for ik, k in enumerate(inder):
            num = std_list[i, k, j]
            fmt = get_appropriate_fmt(num)
            if bin_bf[ik]:
                new_add = f'\\textbf{{$\\pm${num:{fmt}}}}'
            else:
                new_add = f'$\\pm${num:{fmt}}'
            if k == bases[-1]:
                new_add = f'{new_add} & '
            if k == lcpo_v1_ind[-1]:
                new_add = f'{new_add} & '
            str_row += f' & {new_add}'
        str_row = str_row.replace('$\\pm$nan', '')
        str_row = str_row.replace('\\textbf{}', '')
        print(str_row + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')

# ################################################## Buffer Ablation ###################################################
# ################################################### Figure 4 #########################################################

fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
line_st = [
    ([1], 'A2C', 'C0', 1, 'dotted'),
    ([0, 2, 4, 6], 'Best Prescient', 'C7', 1, 'dashed'),
    ([9], 'LCPO $n_b=20M$', 'k', 1, (0, (1, 1))),
    ([10], 'LCPO $n_b=200K$', 'C2', 1, 'solid'),
    ([11], 'LCPO $n_b=500$', 'C3', 1, (0, (3, 1, 1, 1, 1, 1))),
    ([12], 'LCPO $n_b=25$', 'C4', 1, (5, (10, 3))),
]

for o, lbl, col, alph, styl in line_st:
    samp = []
    for j in range(len(env_s)):
        for k in range(len(trl)):
            best = mean_list[j, :, k].max()
            ag = mean_list[j, o, k].max()
            base = mean_list[j, :, k].min()
            assert base < best
            samp.append((ag - base) / (best - base))
    x = np.sort(samp)
    y = np.linspace(0, 100, len(x))
    ax.plot(x, y, label=lbl, color=col, linestyle=styl, alpha=alph)

ax.grid()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
ax.set_xlim(left=0.35, right=1.05)
ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.15, 1.03, 1.2, 0.1), mode="expand", handlelength=3)

fig.supylabel('CDF (\\%)')
fig.supxlabel('Normalized Return')
fig.tight_layout()
plt.subplots_adjust(bottom=0.18, hspace=1.1, wspace=0.16, left=0.15, top=0.77)

plt.savefig('figures/windy_return_buffer.pdf', format='pdf')

# ################################################## Buffer Ablation ###################################################
# #################################################### Table 7 #########################################################

env_names = ['Pendulum', 'Inverse Pendulum', 'Inverse Double Pendulum', 'Hopper', 'Reacher']
labels_lcpo = ['$n_b=20M$', '$n_b=200K$', '$n_b=500$', '$n_b=25$']
labels_bases = ['Online EWC', 'Sliding OGD', 'CLEAR', 'BFDQN', 'MBCD', 'MBPO', 'A2C', 'TRPO', 'DDQN', 'SAC']
labels_oracs = ['A2C', 'TRPO', 'DDQN', 'SAC']

lcpo_ind = [9, 10, 11, 12]
bases = [20, 21, 23, 22, 18, 19, 1, 3, 5, 7]
oracs = [0, 2, 4, 6]

print('\\begin{tabular}{c l ' + ' '.join(['c'] * (len(lcpo_ind) + 2)) + '}')
print('\\toprule')
print(f'& & \\multicolumn{{{len(lcpo_ind)}}}{{c}}{{\\textbf{{LCPO}}}} & & \\\\')
cmid = 3 + len(lcpo_ind) - 1
cend = 3 + len(lcpo_ind) + 1
print(f'\\cmidrule{{3-{cmid}}}')

label_row = f'&'
for k in labels_lcpo:
    label_row += f' & {k}'
label_row += " & Best Baseline & Best Prescient \\\\"
print(label_row)
print()
print('\\midrule')
print()

for i in range(mean_list.shape[0]):
    for j in range(mean_list.shape[-1]):
        data_row = np.array([mean_list[i, k, j] for k in bases])
        base_ind = bases[np.argmax(data_row)]
        base_label = labels_bases[np.argmax(data_row)]

        data_row = np.array([mean_list[i, k, j] for k in oracs])
        orac_ind = oracs[np.argmax(data_row)]
        orac_label = labels_oracs[np.argmax(data_row)]

        inder = lcpo_ind + [base_ind, orac_ind]
        data_row = np.array([mean_list[i, k, j] for k in inder])
        bin_bf = np.zeros(len(inder), dtype=bool)
        bin_bf[np.argmax(data_row[:-1])] = 1

        if j == 0:
            print('\n\\midrule\n')
            str_row = '\\multirow{10}{*}{\\rotatebox[origin=c]{90}{' + env_names[i] + '}} & '
        else:
            print(f'\\cmidrule{{2-{cend}}}')
            str_row = ' & '
        str_row += f'\\multirow{{2}}{{*}}{{Context Trace {j + 1}}}'
        for ik, k in enumerate(inder):
            num = mean_list[i, k, j]
            fmt = get_appropriate_fmt(num)
            if bin_bf[ik]:
                new_add = f'\\textbf{{{num:{fmt}}}}'
            else:
                new_add = f'{num:{fmt}}'
            if k in bases:
                new_add = f'{new_add} ({base_label})'
            if k in oracs:
                new_add = f'\\multirow{{2}}{{*}}{{{new_add} ({orac_label})}}'
            str_row += f' & {new_add}'
        print(str_row + ' \\\\')
        str_row = ' & '
        for ik, k in enumerate(inder):
            num = std_list[i, k, j]
            fmt = get_appropriate_fmt(num)
            if bin_bf[ik]:
                new_add = f'\\textbf{{$\\pm${num:{fmt}}}}'
            else:
                new_add = f'$\\pm${num:{fmt}}'
            if k == bases[-1]:
                new_add = f'{new_add} & '
            str_row += f' & {new_add}'
        str_row = str_row.replace('$\\pm$nan', '')
        str_row = str_row.replace('\\textbf{}', '')
        print(str_row + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')

# ###################################################### Runtime #######################################################
# #################################################### Table 8 #########################################################
labels_all = ['LCPO', 'A2C', 'TRPO', 'DDQN', 'SAC', 'Sliding OGD', 'CLEAR', 'BFDQN', 'MBPO', 'MBCD', 'Online EWC']

lcpo_ind = 8
bases = [1, 3, 5, 7, 21, 23, 22, 19, 18, 20]
inder = [lcpo_ind] + bases

print('\\begin{tabular}{c l ' + ' '.join(['c'] * (len(bases) + 1)) + '}')
print('\\toprule')

label_row = ' & '.join([list_labels[k] for k in inder])
label_row += " \\\\"
print(label_row)
print('\\midrule')

num_list = []
for ik, k in enumerate(inder):
    num = mean_list[k] * 1000 / 200
    if list_labels[k] == 'TRPO':
        num /= 16
    if np.abs(num) >= 100:
        fmt = '.0f'
    elif np.abs(num) >= 10:
        fmt = '.1f'
    else:
        fmt = '.2f'
    num_list += [f'{num:{fmt}}']
print(' & '.join(num_list) + ' \\\\')
num_list = []
for ik, k in enumerate(inder):
    num = std_list[k] * 1000 / 200
    if list_labels[k] == 'TRPO':
        num /= 16
    if np.abs(num) >= 100:
        fmt = '.0f'
    elif np.abs(num) >= 10:
        fmt = '.1f'
    else:
        fmt = '.2f'
    num_list += [f'$\\pm${num:{fmt}}']
    str_row += f' & {new_add}'
print(' & '.join(num_list).replace('$\\pm$nan', '') + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')

# #################################################### EWC Ablation ####################################################
# ##################################################### Load results ###################################################

list_files = []
list_labels = []
tags = [
    'a2c_oracle_eval',
    'trpo_oracle_eval',
    'dqn_oracle_eval',
    'sac_oracle_eval',

    'sac_continual',

    'lcpo_continual',

    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.9998_ewc_alpha0.05',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.9998_ewc_alpha0.1',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.9998_ewc_alpha0.5',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.9998_ewc_alpha1.0',

    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99993_ewc_alpha0.05',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99993_ewc_alpha0.1',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99993_ewc_alpha0.5',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99993_ewc_alpha1.0',

    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99998_ewc_alpha0.05',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99998_ewc_alpha0.1',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99998_ewc_alpha0.5',
    'ewcpp_continual_ablation_env_pend_seed{sd}_ewc_gamma0.99998_ewc_alpha1.0',
]
for tag in tags:
    if 'oracle' in tag:
        list_files.extend([[f'./tests/{tag}_s{sd}_envpend_t{tind}/' for sd in range(num_seeds)] for tind in trl])
    elif 'ewcpp_continual_ablation' in tag:
        list_files.extend(
            [[f'./tests/{tag.format(sd=sd)}_trace_ind{tind}/' for sd in range(num_seeds)] for tind in trl])
    else:
        list_files.extend([[f'./tests/{tag}_seed{sd}_envpend_trace_ind{tind}/' for sd in range(num_seeds)]
                           for tind in trl])
    list_labels.extend([f'{tag}_t{tind}' for tind in trl])

df_list = load_data(list_files, list_labels)

mean_list = np.array([df[df['time'] > 30000]['reward'].mean() for df in df_list]).reshape((len(tags), len(trl)))
std_list = np.array(
    [df[df['time'] > 30000].groupby(by='seed').mean()['reward'].std() * t(0.05, df['seed'].nunique()) for df in
     df_list]).reshape(
    (len(tags), len(trl)))

# #################################################### EWC Ablation ####################################################
# ################################################### Figure 5 #########################################################

fig, axes = plt.subplots(2, 2, figsize=(6.75, 2))

line_st = [
    ([0, 1, 2, 3], '', 'C7', ''),
    ([5], '', 'C2', ''),
    ([4], '', 'C0', ''),
    ([6], '0.05, 1M', 'C9', ''),
    ([7], '0.05, 3M', 'C9', ''),
    ([8], '0.05, 10M', 'C9', ''),
    ([9], '0.1, 1M', 'C9', ''),
    ([10], '0.1, 3M', 'C9', '///'),
    ([11], '0.1, 10M', 'C9', ''),
    ([12], '0.5, 1M', 'C9', ''),
    ([13], '0.5, 3M', 'C9', ''),
    ([14], '0.5, 10M', 'C9', ''),
    ([15], '1.0, 1M', 'C9', ''),
    ([16], '1.0, 3M', 'C9', ''),
    ([17], '1.0, 10M', 'C9', ''),
]

lims = [
    [-180, -230],
    [-330, -630],
    [-175, -470],
    [-330, -430],
]

for i in range(len(trl)):
    labels = []
    cols = []
    means = []
    errs = []
    hs = []
    for o, lbl, cl, hatch in line_st:
        labels.append(lbl)
        cols.append(cl)
        hs.append(hatch)
        if len(o) > 1:
            means.append(mean_list[o, i].max())
            errs.append(0)
        else:
            means.append(mean_list[o[0], i])
            errs.append(std_list[o[0], i])
    row = i % 2
    col = i // 2
    ax = axes[row][col]
    ax.bar(x=np.arange(len(labels)), height=means, yerr=errs, tick_label=labels, color=cols, capsize=1,
           error_kw={'linewidth': 0.5}, hatch=hs)
    ax.tick_params(axis='x', rotation=60)

    if row == 0:
        ax.set_xticklabels(['' for k in ax.get_xticks()])

    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    ax.set_title(f'Context {i + 1}')
    ax.set_ylim(top=lims[i][0], bottom=lims[i][1])

legend_elements = [
    Patch(facecolor='C7', edgecolor='C7', label='Best Prescient'),
    Patch(facecolor='C2', edgecolor='C2', label='LCPO'),
    Patch(facecolor='C0', edgecolor='C0', label='SAC'),
    Patch(facecolor='C9', edgecolor='C9', label='Online EWC'),
]
axes[0][0].legend(handles=legend_elements, ncol=4, loc='lower left', bbox_to_anchor=(0.57, 1.20, 1.0, 0.1),
                  mode="expand")

fig.supylabel('Lifelong Return')
fig.tight_layout()
plt.subplots_adjust(bottom=0.25, hspace=0.4, wspace=0.16, left=0.10, top=0.85)

plt.savefig('figures/windy_ewc_bars.pdf', format='pdf')

# #################################################### EWC Ablation ####################################################
# #################################################### Table 2 #########################################################

env_names = ['Pendulum', 'Inverse Pendulum', 'Inverse Double Pendulum', 'Hopper', 'Reacher']
alpha_vals = ['$\\alpha=0.05$', '$\\alpha=0.1$', '$\\alpha=0.5$', '$\\alpha=1.0$']
beta_vals = ['$1M$', '$3M$', '$10M$']
nbv = len(beta_vals)
ewc_ind = [i for i in range(6, 18)]
assert len(ewc_ind) == len(alpha_vals) * nbv
ncols = len(ewc_ind) + len(ewc_ind) // nbv - 1

print('\\begin{tabular}{l ' + ' '.join(['c'] * ncols) + '}')
print('\\toprule')
print(f'& \\multicolumn{{{ncols}}}{{c}}{{\\textbf{{Online EWC}}}} \\\\')
print(f'\\cmidrule{{2-{2 + ncols - 1}}}')

print('& ' + ' & & '.join([f'\\multicolumn{{{nbv}}}{{c}}{{{a_str}}}' for a_str in alpha_vals]) + '\\\\')
print(' '.join([f'\\cmidrule{{{2 + i * (nbv + 1)}-{2 + i * (nbv + 1) + nbv - 1}}}' for i in range(4)]))

print('$\\beta^{-1}$: & ' + ' & & '.join([' & '.join(beta_vals)] * len(alpha_vals)) + ' \\\\')

print()
print('\\midrule')

for j in range(mean_list.shape[-1]):
    print('\\midrule')
    str_row = f'Context'
    for ik, k in enumerate(ewc_ind):
        num = mean_list[k, j]
        fmt = get_appropriate_fmt(num)
        new_add = f'{num:{fmt}}'
        str_row += f' & {new_add}'
    print(str_row + ' \\\\')
    str_row = f'Trace {j + 1}'
    for ik, k in enumerate(ewc_ind):
        num = std_list[k, j]
        fmt = get_appropriate_fmt(num)
        new_add = f'$\\pm${num:{fmt}}'
        str_row += f' & {new_add}'
    str_row = str_row.replace('$\\pm$nan', '')
    str_row = str_row.replace('\\textbf{}', '')
    print(str_row + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')

# ###################################################### Ideal MBPO ####################################################
# ################################################## Load Results ######################################################

list_files = []
list_labels = []
tags = [
    'a2c_oracle_eval',
    'trpo_oracle_eval',
    'dqn_oracle_eval',
    'sac_oracle_eval',

    'a2c_continual',
    'mbpo_continual',
    'imbpo_continual_env_pend',

    'lcpo_continual',
]
for tag in tags:
    if 'oracle' in tag:
        list_files.extend([[f'./tests/{tag}_s{sd}_envpend_t{tind}/' for sd in range(num_seeds)] for tind in trl])
    elif 'imbpo_continual_env_pend' in tag:
        list_files.extend([[f'./tests/{tag}_seed{sd}_trace_ind{tind}/' for sd in range(num_seeds)] for tind in trl])
    else:
        list_files.extend([[f'./tests/{tag}_seed{sd}_envpend_trace_ind{tind}/' for sd in range(num_seeds)]
                           for tind in trl])
    list_labels.extend([f'{tag}_t{tind}' for tind in trl])

df_list = load_data(list_files, list_labels)

mean_list = np.array([df[df['time'] > 30000]['reward'].mean() for df in df_list]).reshape((len(tags), len(trl)))
std_list = np.array(
    [df[df['time'] > 30000].groupby(by='seed').mean()['reward'].std() * t(0.05, df['seed'].nunique()) for df in
     df_list]).reshape(
    (len(tags), len(trl)))

# ###################################################### Ideal MBPO ####################################################
# #################################################### Table 3 #########################################################

labels_bases = ['LCPO', 'MBPO', 'Ideal MBPO', 'A2C']
labels_oracs = ['A2C', 'TRPO', 'DDQN', 'SAC']

bases = [7, 5, 6, 4]
oracs = [0, 1, 2, 3]

print('\\begin{tabular}{c l ' + ' '.join(['c'] * (len(bases) + 1)) + '}')
print('\\toprule')
print(f'& & \\multicolumn{{{len(bases)}}}{{c}}{{\\textbf{{Online Learning}}}} & \\\\')
print(f'\\cmidrule{{2-{2 + len(bases) - 1}}}')

label_row = f'&'
for k in labels_bases:
    label_row += f' & {k}'
label_row += "& \\textbf{Best Prescient}\\\\"
print(label_row)
print()
print('\\midrule')

for j in range(mean_list.shape[-1]):
    data_row = np.array([mean_list[k, j] for k in oracs])
    orac_ind = oracs[np.argmax(data_row)]
    orac_label = labels_oracs[np.argmax(data_row)]

    inder = bases + [orac_ind]
    bin_bf = np.zeros(len(inder), dtype=bool)
    data_row = np.array([mean_list[k, j] for k in inder])
    bin_bf[np.argmax(data_row[:-1])] = 1
    print('\n\\midrule\n')
    str_row = f'\\multirow{{2}}{{*}}{{Context Trace {j + 1}}}'
    for ik, k in enumerate(inder):
        num = mean_list[k, j]
        fmt = get_appropriate_fmt(num)
        if bin_bf[ik]:
            new_add = f'\\textbf{{{num:{fmt}}}}'
        else:
            new_add = f'{num:{fmt}}'
        if k in oracs:
            new_add = f'\\multirow{{2}}{{*}}{{{new_add} ({orac_label})}}'
        str_row += f' & {new_add}'
    print(str_row + ' \\\\')
    str_row = ' & '
    for ik, k in enumerate(inder):
        num = std_list[k, j]
        fmt = get_appropriate_fmt(num)
        if bin_bf[ik]:
            new_add = f'\\textbf{{$\\pm${num:{fmt}}}}'
        else:
            new_add = f'$\\pm${num:{fmt}}'
        str_row += f' & {new_add}'
    str_row = str_row.replace('$\\pm$nan', '')
    str_row = str_row.replace('\\textbf{}', '')
    print(str_row + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')

# ###################################################### LCPO-P ########################################################
# ################################################## Load Results ######################################################

list_files = []
list_labels = []
tags = [
    'a2c_oracle_eval', 'a2c_continual',
    'trpo_oracle_eval', 'trpo_continual',
    'dqn_oracle_eval', 'dqn_continual',
    'sac_oracle_eval', 'sac_continual',

    'lcpo_continual',

    'lcpo_p_continual_env_pend',

    'mbcd_continual', 'mbpo_continual', 'ewcpp_continual', 'slide_ogd_continual', 'bfdqn_continual', 'clear_continual'
]
for tag in tags:
    if 'oracle' in tag:
        list_files.extend([[f'./tests/{tag}_s{sd}_envpend_t{tind}/' for sd in range(num_seeds)] for tind in trl])
    elif 'lcpo_p_continual_env_pend' in tag:
        list_files.extend([[f'./tests/{tag}_seed{sd}_trace_ind{tind}/' for sd in range(num_seeds)] for tind in trl])
    else:
        list_files.extend([[f'./tests/{tag}_seed{sd}_envpend_trace_ind{tind}/' for sd in range(num_seeds)]
                           for tind in trl])
    list_labels.extend([f'{tag}_t{tind}' for tind in trl])

df_list = load_data(list_files, list_labels)

mean_list = np.array([df[df['time'] > 30000]['reward'].mean() for df in df_list]).reshape((len(tags), len(trl)))
std_list = np.array(
    [df[df['time'] > 30000].groupby(by='seed').mean()['reward'].std() * t(0.05, df['seed'].nunique()) for df in
     df_list]).reshape(
    (len(tags), len(trl)))

# ###################################################### LCPO-P ########################################################
# #################################################### Table 4 #########################################################

labels_lcpo = ['LCPO', 'LCPO-P']
labels_bases = ['Online EWC', 'Sliding OGD', 'CLEAR', 'BFDQN', 'MBCD', 'MBPO', 'A2C', 'TRPO', 'DDQN', 'SAC']
labels_oracs = ['A2C', 'TRPO', 'DDQN', 'SAC']

lcpo_ind = [8, 9]
bases = [12, 13, 15, 14, 10, 11, 1, 3, 5, 7]
oracs = [0, 2, 4, 6]

print('\\begin{tabular}{c l ' + ' '.join(['c'] * (len(lcpo_ind) + 2)) + '}')
print('\\toprule')

label_row = f'&'
for k in labels_lcpo:
    label_row += f' & {k}'
label_row += " & Best Baseline & \textbf{Best Prescient} \\\\"
print(label_row)
print()
print('\\midrule')

for j in range(mean_list.shape[-1]):
    data_row = np.array([mean_list[k, j] for k in bases])
    base_ind = bases[np.argmax(data_row)]
    base_label = labels_bases[np.argmax(data_row)]

    data_row = np.array([mean_list[k, j] for k in oracs])
    orac_ind = oracs[np.argmax(data_row)]
    orac_label = labels_oracs[np.argmax(data_row)]

    inder = lcpo_ind + [base_ind, orac_ind]
    data_row = np.array([mean_list[k, j] for k in inder])
    bin_bf = np.zeros(len(inder)+2, dtype=bool)
    bin_bf[np.argmax(data_row[:-1])] = 1

    print('\\midrule\n')
    str_row = f'\\multirow{{2}}{{*}}{{Context Trace {j + 1}}}'
    for ik, k in enumerate(inder):
        num = mean_list[k, j]
        fmt = get_appropriate_fmt(num)
        if bin_bf[ik]:
            new_add = f'\\textbf{{{num:{fmt}}}}'
        else:
            new_add = f'{num:{fmt}}'
        if k in bases:
            new_add = f'{new_add} ({base_label})'
        if k in oracs:
            new_add = f'\\multirow{{2}}{{*}}{{{new_add} ({orac_label})}}'
        str_row += f' & {new_add}'
    print(str_row + ' \\\\')
    str_row = ' & '
    for ik, k in enumerate(inder):
        num = std_list[k, j]
        fmt = get_appropriate_fmt(num)
        if bin_bf[ik]:
            new_add = f'\\textbf{{$\\pm${num:{fmt}}}}'
        else:
            new_add = f'$\\pm${num:{fmt}}'
        if k == bases[-1]:
            new_add = f'{new_add} & '
        str_row += f' & {new_add}'
    str_row = str_row.replace('$\\pm$nan', '')
    str_row = str_row.replace('\\textbf{}', '')
    print(str_row + ' \\\\')

print('\\bottomrule')
print('\\end{tabular}')
