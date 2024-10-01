[
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type A2C --entropy_max 0 --entropy_decay 0 --entropy_min 0 --master_batch 4608 '
                  '--num_epochs 4500 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'a2c_continual'
    },
    {
        'common': 'python3 para_train_single_trace.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type A2C --entropy_max 0 --entropy_decay 0 --entropy_min 0 --master_batch 128 '
                  '--num_epochs 6000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'a2c_oracle'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type TRPO --trpo_kl_in 1e-2 --trpo_damping 1e-1 '
                  '--entropy_max 0 --entropy_decay 0 --entropy_min 0 '
                  '--master_batch 10240 --num_epochs 2025 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'trpo_continual'
    },
    {
        'common': 'python3 para_train_single_trace.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type TRPO --trpo_kl_in 1e-2 --trpo_damping 1e-1 '
                  '--entropy_max 0 --entropy_decay 0 --entropy_min 0 '
                  '--master_batch 2048 --num_epochs 300 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'trpo_oracle'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type DQN --eps_min 0 --eps_decay 2e-4 --off_policy_tau 0.01 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--entropy_max 0 --entropy_decay 0 --entropy_min 0 '
                  '--master_batch 128 --num_epochs 162000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'dqn_continual'
    },
    {
        'common': 'python3 para_train_single_trace.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type DQN --eps_min 0 --eps_decay 2e-4 --off_policy_tau 0.01 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--entropy_max 0 --entropy_decay 0 --entropy_min 0 '
                  '--master_batch 128 --num_epochs 6000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'dqn_oracle'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type SAC --off_policy_tau 0.005 '
                  '--entropy_max 1e-2 --entropy_decay 0 --entropy_min 0 --auto_target_entropy 0.1 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--master_batch 128 --num_epochs 162000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'sac_continual'
    },
    {
        'common': 'python3 para_train_single_trace.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type SAC --off_policy_tau 0.005 '
                  '--entropy_max 1e-2 --entropy_decay 0 --entropy_min 0 --auto_target_entropy 0.1 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--master_batch 128 --num_epochs 6000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'sac_oracle'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type LCPO --trpo_kl_in 1e-1 --trpo_kl_out 1e-4 --trpo_damping 1e-1 '
                  '--entropy_max 1e-2 --entropy_decay 0 --entropy_min 0 --auto_target_entropy 0.1 '
                  '--master_batch 4608 --num_epochs 4500 --ood_subsample 100 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
            {'arg': 'lcpo_thresh', 'val': ['-5', '-6', '-7']},
        ],
        'name': 'lcpo_continual_ablation'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type SAC-MBCD --off_policy_tau 0.005 '
                  '--entropy_max 1e-2 --entropy_decay 0 --entropy_min 0 --auto_target_entropy 0.1 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--master_batch 128 --num_epochs 162000 --cusum 300000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'mbcd_continual'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type SAC-MBPO --off_policy_tau 0.005 '
                  '--entropy_max 1e-2 --entropy_decay 0 --entropy_min 0 --auto_target_entropy 0.1 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--master_batch 128 --num_epochs 162000 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'mbpo_continual'
    },
    {
        'common': 'python3 train.py --cont_decay 0.81 --lam 0.95 --lr_rate 1e-3 --reward_scale 1600 '
                  '--agent_type EWCPP --off_policy_tau 0.005 '
                  '--entropy_max 1e-2 --entropy_decay 0 --entropy_min 0 --auto_target_entropy 0.1 '
                  '--off_policy_batch_size 512 --off_policy_learn_steps 128 --off_policy_random_epochs 1000 '
                  '--master_batch 128 --num_epochs 162000 --ewc_alpha 0.1 --ewc_gamma 0.99993 ',
        'iterant': [
            {'arg': 'trace_ind', 'val': ['0', '1']},
            {'arg': 'seed', 'val': [f'{s}' for s in range(10)]},
        ],
        'name': 'ewcpp_continual'
    },
]
