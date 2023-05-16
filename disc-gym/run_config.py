[
    {
        'common': 'python3 train.py --agent_type A2C --master_batch 200 --num_epochs 10000 --reward_scale 1 '
                  '--lr_rate 4e-4 --val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 --entropy_max 0 '
                  '--entropy_decay 0 --entropy_min 0',
        'iterant': [
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'a2c_pendulum_disc'
    },
    {
        'common': 'python3 train.py --agent_type TRPO --reward_scale 1 --master_batch 3200 --num_epochs 625 '
                  '--val_lr_rate 1e-3 --n_hid 64 64 --lam 0.9 --gamma 0.99 --entropy_max 0 --eval_interval 10 '
                  '--entropy_decay 0 --entropy_min 0 --trpo_kl_in 1e-2 --trpo_damping 1e-1',
        'iterant': [
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'trpo_pendulum_disc'
    },
    {
        'common': 'python3 train.py --agent_type DQN --master_batch 512 --off_policy_learn_steps 200 '
                  '--num_epochs 10000 --reward_scale 1 --eps_decay 2e-4 --eps_min 0 --gamma 0.99 --n_hid 64 64 '
                  '--lr_rate 1e-3 --off_policy_random_epochs 1000 --off_policy_tau 0.01',
        'iterant': [
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'dqn_pendulum_disc'
    },
    {
        'common': 'python3 train.py --agent_type SAC --master_batch 512 --off_policy_learn_steps 200 '
                  '--num_epochs 10000 --reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000',
        'iterant': [
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'sac_pendulum_disc'
    },
    {
        'common': 'python3 train.py --agent_type MBPO --master_batch 512 --off_policy_learn_steps 200 '
                  '--num_epochs 10000 --reward_scale 1 --entropy_max 1e-1 --entropy_decay 0 --entropy_min 0 '
                  '--gamma 0.99 --n_hid 64 64 --lr_rate 4e-4 --val_lr_rate 1e-3 --auto_target_entropy 0.1 '
                  '--off_policy_tau 0.01 --off_policy_random_epochs 1000 --mbpo_warm_up 0',
        'iterant': [
            {'arg': 'seed', 'val': [f'{i}' for i in range(10)]},
        ],
        'name': 'mbpo_pendulum_disc'
    },
]
