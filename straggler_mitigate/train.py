import pickle
from termcolor import colored
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as AdvActorCritic
from agent.a2c_eval import TrainerNet as AdvActorCriticEval
from agent.lcpo import TrainerNet as LocallyConstPolOpt
from agent.trpo import TrainerNet as TrustRegPolOpt
from agent.sac import TrainerNet as SoftActorCritic
from agent.sac_eval import TrainerNet as SoftActorCriticEval
from agent.sac_mbcd import TrainerNet as MBCDSoftActorCritic
from agent.sac_mbpo import TrainerNet as MBPOSoftActorCritic
from agent.dqn import TrainerNet as DeepQLearning
from agent.dqn_eval import TrainerNet as DeepQLearningEval
from cenv.load_balance import load_balance_env
from param import config
from utils.logger import compact


def start_experiment(agent_type, output_folder):
    if config.compact == 'only' and not config.skip_log:
        compact(output_folder)
        os.remove(f'{output_folder}/log_in_play')
    else:
        os.makedirs(output_folder, exist_ok=True)
        if any(os.path.exists(f'{output_folder}/{file}') for file in ['data.log_ftr', 'data.log_pkl', 'data.log_0',
                                                                      'data.log_agent', 'data.log_all', 'models']):
            print(colored('Log files already exist in output folder', 'red'))
            print(colored('Possibility of overwrite, exiting', 'red'))
            exit(1)
        with open(f'{output_folder}/run_in_play', 'w') as f:
            f.write('started')
        with open(f'{output_folder}/log_in_play', 'w') as f:
            f.write('started')

        if config.device == 'cpu':
            torch.set_num_threads(1)

        args_dict = vars(config)
        args_dict['result_folder'] = output_folder
        args_dict['agent_type'] = agent_type
        with open(f'{output_folder}/args.pkl', 'wb') as handle:
            pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        os.makedirs(output_folder + '/models/', exist_ok=True)

        # set up environments for workers
        print('Setting up environment..')
        env = load_balance_env(output_folder)

        # training monitor
        print('Setting up monitoring..')
        if not config.skip_tb:
            monitor = SummaryWriter(output_folder + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()), flush_secs=10)
        else:
            monitor = None

        # set up trainer
        if agent_type == 'A2C':
            print('Setting up Advantage Actor Critic..')
            agent = AdvActorCritic(env, monitor, output_folder)
        elif agent_type == 'A2C-EVAL':
            print('Setting up Actor Critic Evaluation..')
            agent = AdvActorCriticEval(env, monitor, output_folder)
        elif agent_type == 'LCPO':
            print('Setting up Locally Constrained Policy Optimization ..')
            agent = LocallyConstPolOpt(env, monitor, output_folder)
        elif agent_type == 'TRPO':
            print('Setting up Trust Region Policy Optimization ..')
            agent = TrustRegPolOpt(env, monitor, output_folder)
        elif agent_type == 'SAC':
            print('Setting up Soft Actor Critic..')
            agent = SoftActorCritic(env, monitor, output_folder)
        elif agent_type == 'SAC-EVAL':
            print('Setting up Soft Actor Critic Evaluation..')
            agent = SoftActorCriticEval(env, monitor, output_folder)
        elif agent_type == 'SAC-MBCD':
            print('Setting up MBCD Baseline..')
            agent = MBCDSoftActorCritic(env, monitor, output_folder)
        elif agent_type == 'SAC-MBPO':
            print('Setting up MBPO Baseline..')
            agent = MBPOSoftActorCritic(env, monitor, output_folder)
        elif agent_type == 'DQN':
            print('Setting up Deep Q Learning..')
            agent = DeepQLearning(env, monitor, output_folder)
        elif agent_type == 'DQN-EVAL':
            print('Setting up Deep Q Learning Evaluation..')
            agent = DeepQLearningEval(env, monitor, output_folder)
        else:
            raise ValueError("Don't know this agent")

        print('Start training..')
        agent.run_training()
        env.close()
        del env
        os.remove(f'{output_folder}/run_in_play')
        if config.compact == 'yes' and not config.skip_log:
            compact(output_folder)
            os.remove(f'{output_folder}/log_in_play')


def main():
    start_experiment(config.agent_type, config.result_folder.rstrip('/') + '/')
    print(colored('DONE', 'green'))


if __name__ == '__main__':
    main()
