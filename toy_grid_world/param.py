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
                        choices=['A2C', 'A2C-TAB', 'LCPO'])

    # -- TRPO/LCPO --
    parser.add_argument('--trpo_kl_in', type=float, default=1e-1,
                        help='TRPO - Inner KL divergence limit (default: 0.1)')
    parser.add_argument('--trpo_kl_out', type=float, default=1e-3,
                        help='TRPO - Inner KL divergence limit (default: 0.001)')
    parser.add_argument('--trpo_damping', type=float, default=1e-1,
                        help='TRPO - Conjugate Gradient Damping (default: 0.1)')
    parser.add_argument('--lcpo_thresh', type=float, default=-9, help='LCPO - MVGL threshold (default: -9)')
    parser.add_argument('--trpo_dual', action='store_true', help='TRPO - Solve dual constraint problem')

    config = parser.parse_args()

    if config.val_lr_rate == -1:
        config.val_lr_rate = config.lr_rate

    if config.saved_model and len(config.saved_model) == 1:
        config.saved_model = config.saved_model[0]

    return config
