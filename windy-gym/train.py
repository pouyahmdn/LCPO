import pickle
import numpy as np
from termcolor import colored
import torch
from argparse import Namespace
import time
import os
from torch.utils.tensorboard import SummaryWriter

from agent.a2c import TrainerNet as AdvActorCritic
from agent.sac_mbpo import TrainerNet as ModelBasedPolOpt
from agent.trpo import TrainerNet as TrustRegPolOpt
from agent.ppo import TrainerNet as ProxPolOpt
from agent.a2c_eval import TrainerNet as AdvActorCriticEval
from agent.lcpo import TrainerNet as LocallyConstPolOpt
from agent.lcppo import TrainerNet as LocallyConstProxPolOpt
from agent.dqn import TrainerNet as DeepQLearning
from agent.sac import TrainerNet as SoftActorCritic
from agent.dqn_eval import TrainerNet as DeepQLearningEval
from agent.sac_eval import TrainerNet as SoftActorCriticEval
from agent.sac_mbcd import TrainerNet as MBCDSoftActorCritic
from env.pendulum import WindyPendulum
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
    parallel = agent_type.endswith('-PARA')
    if parallel:
        agent_type = agent_type[:-5]

    print(colored('Running Windy Pendulum', 'red'))
    wind_arr = np.load(f'{config.dataset_folder}/ou_tr{config.trace_ind}.npy')
    wind_arr = np.c_[wind_arr[:len(wind_arr)//2], wind_arr[-len(wind_arr)//2:]]
    print('Wind array shape is ', wind_arr.shape)
    env = WindyPendulum(wind_arr, bins=15, parallel=parallel, threshold=config.lcpo_thresh)
    env_eval = WindyPendulum(wind_arr, bins=15, eval_mode=True, threshold=config.lcpo_thresh)
    eval_len = 100 * 200

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
        len_buff_eval=eval_len,
        parallel=parallel,
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
    elif agent_type == 'TRPO':
        print('Setting up Trust Region Policy Optimization ..')
        agent = TrustRegPolOpt(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            trpo_kl=config.trpo_kl_in,
            trpo_damping=config.trpo_damping,
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
            ood_mini_len=5*config.master_batch,
            ood_len=config.master_batch*config.num_epochs,
        )
    elif agent_type == 'LCPPO':
        print('Setting up Locally Constrained Proximal Policy Optimization ..')
        agent = LocallyConstProxPolOpt(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            ppo_kl=config.ppo_target_kl,
            ppo_iters=config.ppo_iters,
            ppo_clip=config.ppo_epsilon,
            kappa=config.lcppo_kappa,
            ood_mini_len=5*config.master_batch,
            ood_len=config.master_batch*config.num_epochs,
        )
    elif agent_type == 'PPO':
        print('Setting up Proximal Policy Optimization ..')
        agent = ProxPolOpt(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            ppo_kl=config.ppo_target_kl,
            ppo_iters=config.ppo_iters,
            ppo_clip=config.ppo_epsilon,
        )
    elif agent_type == 'ACTOR-EVAL':
        print('Setting up Actor Critic Evaluation..')
        agent = AdvActorCriticEval(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            lam=config.lam,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
        )
    elif agent_type == 'DQN':
        print('Setting up Deep Q Learning..')
        agent = DeepQLearning(
            **common_kwargs,
            eps_decay=config.eps_decay,
            eps_min=config.eps_min,
            num_epochs=config.num_epochs,
            off_policy_random_epochs=config.off_policy_random_epochs,
            off_policy_learn_steps=config.off_policy_learn_steps,
            tau=config.off_policy_tau,
        )
    elif agent_type == 'SAC':
        print('Setting up Soft Actor Critic..')
        agent = SoftActorCritic(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            off_policy_random_epochs=config.off_policy_random_epochs,
            off_policy_learn_steps=config.off_policy_learn_steps,
            tau=config.off_policy_tau,
            num_epochs=config.num_epochs,
        )
    elif agent_type == 'DQN-EVAL':
        print('Setting up DDQN Evaluation..')
        agent = DeepQLearningEval(
            **common_kwargs,
            eps_decay=config.eps_decay,
            eps_min=config.eps_min,
            num_epochs=config.num_epochs,
            off_policy_random_epochs=config.off_policy_random_epochs,
            off_policy_learn_steps=config.off_policy_learn_steps,
            tau=config.off_policy_tau,
        )
    elif agent_type == 'SAC-EVAL':
        print('Setting up Soft Actor Critic Evaluation..')
        agent = SoftActorCriticEval(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            off_policy_random_epochs=config.off_policy_random_epochs,
            off_policy_learn_steps=config.off_policy_learn_steps,
            tau=config.off_policy_tau,
            num_epochs=config.num_epochs,
        )
    elif agent_type == 'MBCD':
        print('Setting up MBCD Baseline..')
        agent = MBCDSoftActorCritic(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            off_policy_random_epochs=config.off_policy_random_epochs,
            off_policy_learn_steps=config.off_policy_learn_steps,
            tau=config.off_policy_tau,
            num_epochs=config.num_epochs,
            cusum=config.mbcd_cusum,
        )
    elif agent_type == 'MBPO':
        print('Setting up Model Based Policy Optimization..')
        agent = ModelBasedPolOpt(
            **common_kwargs,
            entropy_max=config.entropy_max,
            entropy_min=config.entropy_min,
            entropy_decay=config.entropy_decay,
            val_lr_rate=config.val_lr_rate,
            auto_target_entropy=config.auto_target_entropy,
            ent_lr=config.ent_lr,
            off_policy_random_epochs=config.off_policy_random_epochs,
            off_policy_learn_steps=config.off_policy_learn_steps,
            tau=config.off_policy_tau,
            num_epochs=config.num_epochs,
            mbpo_warm_up=config.mbpo_warm_up,
            use_oracle_mbpo=config.use_oracle_mbpo,
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
