import pickle
from termcolor import colored
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter

from agent.para_a2c import TrainerNet as ParaAdvActorCritic
from agent.para_trpo import TrainerNet as ParaTrustRegPol
from agent.para_sac import TrainerNet as ParaSoftActorCritic
from agent.para_dqn import TrainerNet as ParaDeepQLearning
from train import start_experiment as start_experiment_eval
from cenv.load_balance import load_balance_env_multi
from param import config
from utils.logger import compact


def start_experiment(agent_type, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    if any(os.path.exists(f'{output_folder}/{file}') for file in ['data.log_ftr', 'data.log_pkl', 'data.log_0',
                                                                  'data.log_agent', 'data.log_all', 'models']):
        print(colored('Log files already exist in output folder', 'red'))
        print(colored('Possibility of overwrite, exiting', 'red'))
        return
    with open(f'{output_folder}/run_in_play', 'w') as f:
        f.write('started')
    with open(f'{output_folder}/log_in_play', 'w') as f:
        f.write('started')

    if config.device == 'cpu':
        torch.set_num_threads(2)

    args_dict = vars(config)
    args_dict['result_folder'] = output_folder
    args_dict['agent_type'] = agent_type
    with open(f'{output_folder}/args.pkl', 'wb') as handle:
        pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(output_folder + '/models/', exist_ok=True)

    # set up environments for workers
    print('Setting up environments..')
    env_s = load_balance_env_multi(output_folder, True, 4)

    # training monitor
    print('Setting up monitoring..')
    monitor = SummaryWriter(output_folder + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()), flush_secs=10)

    # set up trainer
    if agent_type == 'A2C':
        print('Setting up Advantage Actor Critic..')
        agent = ParaAdvActorCritic(env_s, monitor, output_folder)
    elif agent_type == 'TRPO':
        print('Setting up Advantage Actor Critic..')
        agent = ParaTrustRegPol(env_s, monitor, output_folder)
    elif agent_type == 'DQN':
        print('Setting up Deep Q Learning..')
        agent = ParaDeepQLearning(env_s, monitor, output_folder)
    elif agent_type == 'SAC':
        print('Setting up Soft Actor Critic..')
        agent = ParaSoftActorCritic(env_s, monitor, output_folder)
    else:
        raise ValueError("Don't know this agent")

    print('Start training..')
    agent.run_training()
    for env in env_s:
        env.close()
    os.remove(f'{output_folder}/run_in_play')


def main():
    start_experiment(config.agent_type, config.result_folder.rstrip('/') + '/')
    print(colored('DONE', 'green'))

    old_path = config.result_folder.rstrip('/')

    path_eval = config.result_folder.rstrip('/').split('/')
    path_eval[-1] = 'eval_' + path_eval[-1]
    path_eval = '/'.join(path_eval) + '/'

    path_model = config.result_folder.rstrip('/') + f'/models/model_{config.num_epochs}'
    assert os.path.exists(path_model)

    config.master_batch = 128
    config.saved_model = path_model
    config.num_epochs = 162000

    if config.agent_type in ['A2C', 'TRPO']:
        start_experiment_eval('A2C-EVAL', path_eval)
    elif config.agent_type == 'SAC':
        start_experiment_eval('SAC-EVAL', path_eval)
    elif config.agent_type == 'DQN':
        start_experiment_eval('DQN-EVAL', path_eval)
    else:
        raise ValueError

    compact(path_eval)
    print(colored('DONE EVAL', 'green'))
    os.remove(f'{old_path}/log_in_play')


if __name__ == '__main__':
    main()
