import argparse


def get_params():
    parser = argparse.ArgumentParser(description='parameters')

    # -- Basic --
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--eps', type=float, default=1e-6, help='epsilon (default: 1e-6)')
    parser.add_argument('--device', type=str, default='cpu', help='torch device')
    parser.add_argument('--skip_tb', action='store_true', help='Skip Tensorboard logging')

    # -- Learning --
    parser.add_argument('--saved_model', type=str, nargs='*', default=None, help='path for saved model (default: None)')
    parser.add_argument('--result_folder', type=str, required=True, help='path for results')
    parser.add_argument('--lr_rate', type=float, default='4e-4', help='learning rate (default: 4e-4)')
    parser.add_argument('--val_lr_rate', type=float, default=-1, help='value learning rate (default: inactive)')
    parser.add_argument('--entropy_max', type=float, default=0.1,
                        help='Max Entropy ratio during PG training (default: 0.1)')
    parser.add_argument('--entropy_min', type=float, default=0,
                        help='Min Entropy ratio during PG training (default: 0)')
    parser.add_argument('--entropy_decay', type=float, default=2e-5,
                        help='Entropy ratio decay during PG training (default: 2e-5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='td lambda factor (default: 0.95)')
    parser.add_argument('--num_epochs', type=int, default=100000,
                        help='number of epochs (default: 100000)')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='model saving interval (default: 100)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='model evaluation interval (default: 1000)')
    parser.add_argument('--n_hid', type=int, default=[256, 256],
                        nargs='+', help='hidden neurons (default: [256, 256])')
    parser.add_argument('--master_batch', type=int, default=10240,
                        help='master batch size to learn (default: 10240)')
    parser.add_argument('--reward_scale', type=float, default='1',
                        help='reward normalization scale (default: 1)')

    parser.add_argument('--ent_lr', type=float, default=1e-3,
                        help='Entropy tuning learning rate (default: 1e-3)')
    parser.add_argument('--auto_target_entropy', type=float, default=-1,
                        help='Auto tune entropy - target (default: -1(off))')

    # -- RLType --
    parser.add_argument('--agent_type', type=str, required=True, help='Agent type (required)',
                        choices=['A2C', 'TRPO', 'PPO', 'DQN', 'SAC', 'MBPO'])

    # -- DQN --
    parser.add_argument('--off_policy_buffer_size', type=int, default=int(1e6),
                        help='Buffer size for off policy method (default: 1e5)')
    parser.add_argument('--off_policy_learn_steps', type=int, default=128,
                        help='Epoch length in steps (default: 128)')
    parser.add_argument('--off_policy_batch_size', type=int, default=8192,
                        help='Epoch length in steps (default: 8192)')
    parser.add_argument('--off_policy_random_epochs', type=int, default=500,
                        help='Initial random period of off policy method (default: 1e5)')
    parser.add_argument('--off_policy_tau', type=float, default=0.01,
                        help='Tau value for soft copy (default: 0.01)')
    parser.add_argument('--eps_min', type=float, default=0, help='Min Eps random exploration during QL (default: 0)')
    parser.add_argument('--eps_decay', type=float, default=4e-5,
                        help='Eps exploration decay rate during QL (default: 4e-5)')

    # -- PPO --
    parser.add_argument('--ppo_iters', type=int, default=30, help='PPO - number of training iterations (default: 30)')
    parser.add_argument('--ppo_epsilon', type=float, default=0.2, help='PPO - epsilon value (default: 0.2)')
    parser.add_argument('--ppo_target_kl', type=float, default=0.01, help='PPO - target KL divergence (default: 0.01)')

    # -- TRPO --
    parser.add_argument('--trpo_kl_in', type=float, default=1e-1,
                        help='TRPO - Inner KL divergence limit (default: 0.1)')
    parser.add_argument('--trpo_damping', type=float, default=1e-1,
                        help='TRPO - Conjugate Gradient Damping (default: 0.1)')

    # -- MBPO --
    parser.add_argument('--mbpo_warm_up', type=int, default=1000, help='MBPO warm up epochs')

    config = parser.parse_args()

    if config.val_lr_rate == -1:
        config.val_lr_rate = config.lr_rate

    if config.saved_model and len(config.saved_model) == 1:
        config.saved_model = config.saved_model[0]

    return config
