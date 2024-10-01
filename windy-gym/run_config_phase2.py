[
    {
        'common': 'python3 train.py --agent_type ACTOR-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --lam 0.9 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'env', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{env}', f'{trace}',
                         f'./tests/a2c_oracle_model_seed{seed}_env{env}_trace_ind{trace}/models/model_10000', f'{ne}')
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_env{env}_t{trace}'
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace in range(4)]
            },
        ],
        'name': 'a2c_oracle_eval'
    },
    {
        'common': 'python3 train.py --agent_type DQN-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'env', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{env}', f'{trace}',
                         f'./tests/dqn_oracle_model_seed{seed}_env{env}_trace_ind{trace}/models/model_10000', f'{ne}')
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_env{env}_t{trace}'
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace in range(4)]
            },
        ],
        'name': 'dqn_oracle_eval'
    },
    {
        'common': 'python3 train.py --agent_type ACTOR-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --lam 0.9 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'env', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{env}', f'{trace}',
                         f'./tests/trpo_oracle_model_seed{seed}_env{env}_trace_ind{trace}/models/model_1000', f'{ne}')
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_env{env}_t{trace}'
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace in range(4)]
            },
        ],
        'name': 'trpo_oracle_eval'
    },
    {
        'common': 'python3 train.py --agent_type SAC-EVAL --master_batch 200 --reward_scale 1 '
                  '--n_hid 64 64 --gamma 0.99',
        'iterant': [
            {
                'arg': ('seed', 'env', 'trace_ind', 'saved_model', 'num_epochs'),
                'val': [(f'{seed}', f'{env}', f'{trace}',
                         f'./tests/sac_oracle_model_seed{seed}_env{env}_trace_ind{trace}/models/model_10000', f'{ne}')
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace, ne in zip(range(4), [100000] * 2 + [40000] * 2)],
                'rep': [f's{seed}_env{env}_t{trace}'
                        for seed in range(5)
                        for env in ['pend', 'inv_pend', 'inv_d_pend', 'reacher', 'hopper']
                        for trace in range(4)]
            },
        ],
        'name': 'sac_oracle_eval'
    },
]
