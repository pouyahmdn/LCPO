# NOTE:
# As observed  in the run setup below, we are not using random seeds.
#
# We originally tried many random seeds for this experiment, but observed instability due to premature exploitation with
# a portion of the seeds. In other words, agents had a hard time exploring after mode changes.
# The point of this toy example is to explain the main idea behind LCPO, catastrophic forgetting and tabular RL, and is
# not supposed to be an evaluation of LCPO vs baselines. As such, we removed all seeds where exploration was a
# hindrance to the main point.
#
# We DO NOT do this for the main evaluation. All main evaluation runs (LCPO, baselines, in both environments) use 10
# seeds (0 to 9).
#
[
    {
        'common': 'python3 train.py --agent_type LCPO --reward_scale 1 --auto_target_entropy 0.1 --eval_interval 10 '
                  '--val_lr_rate 1e-3 --n_hid 32 32 --lam 0.8 --gamma 0.9 --master_batch 200 --num_epochs 20000 '
                  '--entropy_decay 0 --entropy_min 0 --trpo_kl_in 1e-1 --trpo_kl_out 1e-5 --trpo_damping 1e-1 '
                  '--entropy_max 1e-1',
        'iterant': [
            {'arg': 'seed', 'val': ['0', '2', '3', '4']},
        ],
        'name': 'lcpo_grid'
    },
    {
        'common': 'python3 train.py --agent_type A2C --master_batch 200 --num_epochs 20000 --reward_scale 1 '
                  '--lr_rate 1e-3 --val_lr_rate 1e-3 --n_hid 32 32 --lam 0.8 --gamma 0.9 --entropy_max 0 '
                  '--entropy_decay 0 --entropy_min 0 --eval_interval 50',
        'iterant': [
            {'arg': 'seed', 'val': ['8', '15', '16', '19']},

        ],
        'name': 'a2c_grid'
    },
    {
        'common': 'python3 train.py --agent_type A2C-TAB --master_batch 200 --num_epochs 20000 --reward_scale 1 '
                  '--lr_rate 1e-1 --val_lr_rate 1e-3 --n_hid 32 32 --lam 0.95 --gamma 0.9 --entropy_max 0 '
                  '--entropy_decay 0 --entropy_min 0 --eval_interval 50',
        'iterant': [
            {'arg': 'seed', 'val': ['5', '8', '11', '21']},

        ],
        'name': 'a2c_tab_grid'
    },
]
