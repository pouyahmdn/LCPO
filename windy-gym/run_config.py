[
    {
        'common': 'python3 train.py --agent_type A2C-PARA --master_batch 3200 --num_epochs 10000 --reward_scale 1 '
                  '--entropy_max 0 --entropy_decay 0 --entropy_min 0 --gamma 0.99 --lam 0.9 --n_hid 64 64 '
                  '--lr_rate 4e-4 --val_lr_rate 1e-3',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': 'trace_ind', 'val': [f'{trace}' for trace in range(4)]},
        ],
        'name': 'a2c_oracle_model'
    },
    {
        'common': 'python3 train.py --agent_type DQN-PARA --master_batch 512 --off_policy_learn_steps 3200 '
                  '--num_epochs 10000 --reward_scale 1 --eps_decay 2e-4 --eps_min 0 --gamma 0.99 --n_hid 64 64 '
                  '--lr_rate 1e-3 --off_policy_random_epochs 1000 --off_policy_tau 0.01',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': 'trace_ind', 'val': [f'{trace}' for trace in range(4)]},
        ],
        'name': 'dqn_oracle_model'
    },
    {
        'common': 'python3 train.py --agent_type TRPO-PARA --master_batch 10000 --num_epochs 1000 --reward_scale 1 '
                  '--entropy_max 0 --entropy_decay 0 --entropy_min 0 --gamma 0.99 --lam 0.9 --n_hid 64 64 '
                  '--trpo_kl_in 1e-2 --trpo_damping 1e-1 --val_lr_rate 1e-3 --eval_interval 10',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': 'trace_ind', 'val': [f'{trace}' for trace in range(4)]},
        ],
        'name': 'trpo_oracle_model'
    },
    {
        'common': 'python3 train.py --agent_type SAC-PARA --master_batch 512 --off_policy_learn_steps 3200 '
                  '--num_epochs 10000 --reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': 'trace_ind', 'val': [f'{trace}' for trace in range(4)]},
        ],
        'name': 'sac_oracle_model'
    },
    {
        'common': 'python3 train.py --agent_type LCPO --reward_scale 1 --auto_target_entropy 0.1 --lcpo_thresh 1 '
                  '--val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 '
                  '--entropy_decay 0 --entropy_min 0 --trpo_kl_in 1e-1 --trpo_kl_out 1e-4 --trpo_damping 1e-1 '
                  '--master_batch 200 --eval_interval 100 --entropy_max 3e-2',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'lcpo_continual'
    },
    {
        'common': 'python3 train.py --agent_type LCPO --reward_scale 1 --auto_target_entropy 0.1 --lcpo_thresh 1 '
                  '--val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 --trpo_dual '
                  '--entropy_decay 0 --entropy_min 0 --trpo_kl_in 1e-1 --trpo_kl_out 1e-4 --trpo_damping 1e-1 '
                  '--master_batch 200 --eval_interval 100 --entropy_max 3e-2',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'lcpo_d_continual'
    },
    {
        'common': 'python3 train.py --agent_type LCPPO --reward_scale 1 --auto_target_entropy 0.1 --lcpo_thresh 1 '
                  '--val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 '
                  '--entropy_decay 0 --entropy_min 0 --ppo_epsilon 0.2 --ppo_target_kl 0.01 --ppo_iters 30 '
                  '--master_batch 200 --eval_interval 100 --entropy_max 3e-2 --lcppo_kappa 10',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'lcpp_p_continual'
    },
    {
        'common': 'python3 train.py --agent_type A2C --master_batch 200 --reward_scale 1 '
                  '--lr_rate 4e-4 --val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 --entropy_max 0 '
                  '--entropy_decay 0 --entropy_min 0',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'a2c_continual'
    },
    {
        'common': 'python3 train.py --agent_type TRPO --reward_scale 1 --master_batch 3200 '
                  '--val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 --entropy_max 0 --eval_interval 10 '
                  '--entropy_decay 0 --entropy_min 0 --trpo_kl_in 1e-2 --trpo_damping 1e-1',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '6250'), ('1', '6250'), ('2', '2500'), ('3', '2500')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'trpo_continual'
    },
    {
        'common': 'python3 train.py --agent_type DQN --master_batch 512 --off_policy_learn_steps 200 '
                  '--reward_scale 1 --eps_decay 2e-4 --eps_min 0 --gamma 0.99 --n_hid 64 64 '
                  '--lr_rate 1e-3 --off_policy_random_epochs 1000 --off_policy_tau 0.01',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'dqn_continual'
    },
    {
        'common': 'python3 train.py --agent_type SAC --master_batch 512 --off_policy_learn_steps 200 '
                  '--reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'sac_continual'
    },
    {
        'common': 'python3 train.py --agent_type MBCD --master_batch 512 --off_policy_learn_steps 200 '
                  '--reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000 --mbcd_cusum 1000',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'mbcd_continual'
    },
    {
        'common': 'python3 train.py --agent_type MBPO --master_batch 512 --off_policy_learn_steps 200 '
                  '--reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000 --mbpo_warm_up 0',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'mbpo_continual'
    },
    {
        'common': 'python3 train.py --agent_type MBPO --master_batch 512 --off_policy_learn_steps 200 '
                  '--reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000 --mbpo_warm_up 0 --use_oracle_mbpo',
        'iterant': [
            {'arg': 'seed', 'val': [f'{s}' for s in range(1)]},
            {'arg': ('trace_ind', 'num_epochs'), 'val': [('0', '100000'), ('1', '100000'), ('2', '40000'), ('3', '40000')],
             'rep': ['trace_ind0', 'trace_ind1', 'trace_ind2', 'trace_ind3']},
        ],
        'name': 'imbpo_continual'
    },
    {
        'common': 'python3 train.py --agent_type ACTOR-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --lam 0.9 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{trace}', f'./tests/a2c_oracle_model_seed{seed}_trace_ind{trace}/models/model_10000', f'{ne}')
                        for seed in range(1)
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_t{trace}' for seed in range(1) for trace in range(4)]
            },
        ],
        'name': 'a2c_oracle_eval'
    },
    {
        'common': 'python3 train.py --agent_type DQN-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{trace}', f'./tests/dqn_oracle_model_seed{seed}_trace_ind{trace}/models/model_10000', f'{ne}')
                        for seed in range(1)
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_t{trace}' for seed in range(1) for trace in range(4)]
            },
        ],
        'name': 'dqn_oracle_eval'
    },
    {
        'common': 'python3 train.py --agent_type ACTOR-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --lam 0.9 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{trace}', f'./tests/trpo_oracle_model_seed{seed}_trace_ind{trace}/models/model_1000', f'{ne}')
                        for seed in range(1)
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_t{trace}' for seed in range(1) for trace in range(4)]
            },
        ],
        'name': 'trpo_oracle_eval'
    },
    {
        'common': 'python3 train.py --agent_type SAC-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{trace}', f'./tests/sac_oracle_model_seed{seed}_trace_ind{trace}/models/model_10000', f'{ne}')
                        for seed in range(1)
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_t{trace}' for seed in range(1) for trace in range(4)]
            },
        ],
        'name': 'sac_oracle_eval'
    },
]
