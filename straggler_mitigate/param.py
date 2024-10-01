import argparse

parser = argparse.ArgumentParser(description='parameters')

# -- Basic --
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--device', type=str, default='cpu', help='torch device')

# -- Load balance --
parser.add_argument('--num_servers', type=int, default=10, help='number of servers (default: 10)')
parser.add_argument('--lb_timeout_levels', type=float, default=[3, 10, 30, 60, 100, 300, 600, 1000, -1], nargs='+',
                    help='timeout levels for resending requests (default: [3, 10, 30, 60, 100, 300, 600, 1000, -1])')
parser.add_argument('--dataset_folder', type=str, default='./traces/', help='The trace dataset folder')
parser.add_argument('--trace_ind', type=int, default=0, help='Trace index to choose')
parser.add_argument('--start_time', type=float, default=0, help='Starting time in trace (default: 0)')
parser.add_argument('--time_window', type=float, default=0.5, help='Time window of actions (default: 0.5)')
parser.add_argument('--max_num_retries', type=int, default=1, help='maximum number of retries (default: 1)')
parser.add_argument('--extra_multiply_penalty', type=float, default=3, help='extra multiply penalty (default: 3)')

parser.add_argument('--tw_safe_queue_size', type=int, default=50, help='safe size of max queue (default: 50)')
parser.add_argument('--tw_exit_queue_size', type=int, default=3, help='training wheel exit queue size (default: 3)')

parser.add_argument('--compact', type=str, default='yes', choices=['yes', 'no', 'only'], help='Agent type (required)')
parser.add_argument('--skip_log', action='store_true', help='Skip logging')
parser.add_argument('--skip_tb', action='store_true', help='Skip tensorboard')

# -- General RL --
parser.add_argument('--agent_type', type=str, required=True, help='Agent type (required)',
                    choices=['A2C', 'A2C-EVAL', 'DQN', 'DQN-EVAL', 'SAC', 'SAC-EVAL', 'LCPO', 'TRPO', 'SAC-MBCD',
                             'SAC-MBPO', 'EWCPP'])
parser.add_argument('--saved_model', type=str, nargs='*', default=None, help='path for saved model (default: None)')
parser.add_argument('--result_folder', type=str, default='./tests/', help='path for results (default: ./tests/)')
parser.add_argument('--lr_rate', type=float, required=True, help='learning rate (default: 1e-3)')
parser.add_argument('--val_lr_rate', type=float, default=-1, help='value learning rate (default: inactive)')
parser.add_argument('--num_epochs', type=int, default=100000, help='number of epochs (default: 100000)')
parser.add_argument('--save_interval', type=int, default=100, help='model saving interval (default: 100)')
parser.add_argument('--cont_decay', type=float, required=True, help='gamma value per second')
parser.add_argument('--nn_map', type=int, default=[32, 16], nargs='+',
                    help='Perm Invar - Mapper hidden layers (default: [32, 16])')
parser.add_argument('--nn_red', type=int, default=[32, 16], nargs='+',
                    help='Perm Invar - Reducer hidden layers (default: [32, 16])')

# -- A2C --
parser.add_argument('--entropy_max', type=float, default=0.1,
                    help='Max Entropy ratio during PG training (default: 0.1)')
parser.add_argument('--entropy_min', type=float, default=0,
                    help='Min Entropy ratio during PG training (default: 0)')
parser.add_argument('--entropy_decay', type=float, default=2e-5,
                    help='Entropy ratio decay during PG training (default: 2e-5)')
parser.add_argument('--lam', type=float, default=0.95,
                    help='td lambda factor (default: 0.95)')
parser.add_argument('--master_batch', type=int, default=128,
                    help='master batch size to learn (default: 128)')
parser.add_argument('--reward_scale', type=float, required=True,
                    help='reward normalization scale (default: 1000)')

# -- DDQN/SAC --
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
parser.add_argument('--eps_decay', type=float, default=4e-5, help='Eps exploration decay rate during QL (default: 4e-5)')

# -- SAC/LCPO --
parser.add_argument('--ent_lr', type=float, default=1e-3, help='Entropy tuning learning rate (default: 1e-3)')
parser.add_argument('--auto_target_entropy', type=float, default=None, help='Auto tune entropy - target (default: off)')

# -- TRPO/LCPO --
parser.add_argument('--trpo_kl_in', type=float, default=1e-1, help='TRPO - Inner KL divergence limit (default: 0.1)')
parser.add_argument('--trpo_kl_out', type=float, default=1e-3, help='TRPO - Inner KL divergence limit (default: 0.001)')
parser.add_argument('--trpo_damping', type=float, default=1e-1, help='TRPO - Conjugate Gradient Damping (default: 0.1)')

# -- LCPO --
parser.add_argument('--lcpo_thresh', type=float, default=-3, help='LCPO threshold (default: -3)')
parser.add_argument('--ood_subsample', type=int, default=100, help='OOD capacity scale down factor (default: 100)')

# -- MBCD --
parser.add_argument('--cusum', type=float, default=1000, help='CUSUM for MBCD')

# -- EWCPP --
parser.add_argument('--ewc_alpha', type=float, default=0.1, help='EWC alpha')
parser.add_argument('--ewc_gamma', type=float, default=0.99993, help='EWC alpha')

config = parser.parse_args()

if config.val_lr_rate == -1:
    config.val_lr_rate = config.lr_rate
