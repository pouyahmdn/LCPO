import pickle
from termcolor import colored
import torch
from argparse import Namespace
import time
import os
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as AdvActorCritic
from agent.a2c_tabular import TrainerNet as TabAdvActorCritic
from agent.lcpo import TrainerNet as LocallyConstPolOpt
from env.grid import ShakyGrid
from param import get_params
from utils.tb_to_pd import tabulate_events


def start_experiment(agent_type: str, output_folder: str, config: Namespace):
    os.makedirs(output_folder, exist_ok=True)
    if any(os.path.exists(f'{output_folder}/{file}') for file in ['models', 'tb.pkl']):
        print(colored('Log files already exist in output folder', 'red'))
        print(colored('Possibility of overwrite, exiting', 'red'))
        exit(1)
    with open(f'{output_folder}/run_in_play', 'w') as f:
        f.write('started train')
    with open(f'{output_folder}/log_in_play', 'w') as f:
        f.write('started log')

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

    print(colored('Running Shaky Grid', 'red'))
    env = ShakyGrid(eval_mode=False)
    env_eval = ShakyGrid(eval_mode=True)
    eval_len = len(env_eval.eval_loop) * env.trunc_len

    # training monitor
    print('Setting up monitoring..')
    if not config.skip_tb:
        monitor = SummaryWriter(output_folder + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()), flush_secs=10)
    else:
        monitor = None

    common_kwargs = dict(
        environment=env,
        environment_eval=env_eval,
        monitor=monitor,
        output_folder=output_folder,
        device=config.device,
        seed=config.seed,
        nn_hids=config.n_hid,
        batch_size=config.master_batch,
        reward_scale=config.reward_scale,
        lr_rate=config.lr_rate,
        gamma=config.gamma,
        len_buff_eval=eval_len
    )

    # set up trainer
    if agent_type == 'A2C':
        print('Setting up Advantage Actor Critic..')
        agent = AdvActorCritic(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
        )
    elif agent_type == 'A2C-TAB':
        print('Setting up Tabular Advantage Actor Critic..')
        agent = TabAdvActorCritic(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
        )
    elif agent_type == 'LCPO':
        print('Setting up Locally Constrained Policy Optimization ..')
        agent = LocallyConstPolOpt(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            trpo_kl_in=config.trpo_kl_in,
            trpo_kl_out=config.trpo_kl_out,
            trpo_damping=config.trpo_damping,
            trpo_dual=config.trpo_dual,
            ood_mini_len=config.master_batch,
            ood_len=config.master_batch*config.num_epochs,
        )
    else:
        raise ValueError("Don't know this agent")

    print('Start training..')
    agent.run_training(
        saved_model=config.saved_model,
        num_epochs=config.num_epochs,
        skip_tb=config.skip_tb,
        save_interval=config.save_interval,
        eval_interval=config.eval_interval,
    )
    env.close()
    env_eval.close()
    del env, env_eval
    os.remove(f'{output_folder}/run_in_play')

    if not config.skip_tb:
        tabulate_events(f'{output_folder}/')
    os.remove(f'{output_folder}/log_in_play')


def main():
    config = get_params()
    start_experiment(config.agent_type, config.result_folder.rstrip('/') + '/', config)
    print(colored('DONE', 'green'))


if __name__ == '__main__':
    main()
