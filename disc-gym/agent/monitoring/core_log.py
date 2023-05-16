import numpy as np
from torch.utils.tensorboard import SummaryWriter

from buffer.buffer import TransitionBuffer
from neural_net.nn import FullyConnectNN, FCNPolicy
from utils.proj_time import ProjectFinishTime


def log_a2c(buff: TransitionBuffer, policy_net: FCNPolicy, ret_np: np.ndarray,
            v_np: np.ndarray, adv_np: np.ndarray, pg_loss: float, v_loss: float, entropy_factor: float,
            norm_entropy: float, log_pi_min: float, elapsed: float, monitor: SummaryWriter,
            proj_eta: ProjectFinishTime, epoch: int, index_model: int):
    avg_reward = log_stats_core(buff, elapsed, monitor, epoch)

    # gather statistics
    ret_mean = ret_np.mean()
    v_net_mean = v_np.mean()
    vf_var = 1-(v_np - ret_np).var()/ret_np.var()
    adv_mean = adv_np.mean()

    # monitor statistics
    monitor.add_scalar('Loss/pg_loss', pg_loss, epoch)
    monitor.add_scalar('Loss/v_loss', v_loss, epoch)
    monitor.add_scalar('Loss/vf_explaned_var', vf_var, epoch)
    monitor.add_scalar('Loss/value_target', ret_mean, epoch)
    monitor.add_scalar('Loss/adv_mean', adv_mean, epoch)
    monitor.add_scalar('Loss/value_network', v_net_mean, epoch)

    monitor.add_scalar('Policy/entropy_factor', entropy_factor, epoch)
    monitor.add_scalar('Policy/norm_entropy', norm_entropy, epoch)
    monitor.add_scalar('Policy/log_pi_min', log_pi_min, epoch)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 3) +
          ' reward: {0:.{1}f},'.format(avg_reward, 2) +
          ' model: {0},'.format(index_model) +
          ' v loss: {0:.{1}f},'.format(v_loss, 3) +
          ' VFEVAR: {0:.{1}f},'.format(vf_var, 3) +
          ' v out: {0:.{1}f},'.format(v_net_mean, 2) +
          ' ret out: {0:.{1}f},'.format(ret_mean, 2) +
          ' entropy: {0:.{1}f}'.format(norm_entropy, 2))


def log_sac(buff: TransitionBuffer,
            q_target: np.ndarray, q_local_1: np.ndarray, q_local_2: np.ndarray, pg_loss: float, q_loss_1: float,
            q_loss_2: float, entropy_factor: float, norm_entropy: float, elapsed: float, monitor: SummaryWriter,
            proj_eta: ProjectFinishTime, epoch: int):
    avg_reward = log_stats_core(buff, elapsed, monitor, epoch)

    q1_var = 1-(q_local_1 - q_target).var()/q_target.var()
    q2_var = 1-(q_local_2 - q_target).var()/q_target.var()
    ql1_avg = q_local_1.mean()
    ql2_avg = q_local_2.mean()
    qt_avg = q_target.mean()

    # monitor statistics
    monitor.add_scalar('Loss/pg_loss', pg_loss, epoch)
    monitor.add_scalar('Loss/v_loss_1', q_loss_1, epoch)
    monitor.add_scalar('Loss/v_loss_2', q_loss_2, epoch)
    monitor.add_scalar('Loss/q_target', qt_avg, epoch)
    monitor.add_scalar('Loss/q_local_1', ql1_avg, epoch)
    monitor.add_scalar('Loss/q_local_2', ql2_avg, epoch)
    monitor.add_scalar('Loss/qf1_explaned_var', q1_var, epoch)
    monitor.add_scalar('Loss/qf2_explaned_var', q2_var, epoch)

    monitor.add_scalar('Policy/entropy_factor', entropy_factor, epoch)
    monitor.add_scalar('Policy/norm_entropy', norm_entropy, epoch)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 3) +
          ' reward: {0:.{1}f},'.format(avg_reward, 2) +
          ' pg loss: {0:.{1}f},'.format(pg_loss, 3) +
          ' q loss: {0:.{2}f} {1:.{2}f},'.format(q_loss_1, q_loss_2, 3) +
          ' q out: {0:.{2}f} {1:.{2}f},'.format(ql1_avg, ql2_avg, 2) +
          ' target out: {0:.{1}f}'.format(qt_avg, 2) +
          ' entropy: {0:.{1}f}'.format(norm_entropy, 2))


def log_dqn(buff: TransitionBuffer, q_np: np.ndarray,
            trg_np: np.ndarray, v_loss: float, epsilon: float, elapsed: float, monitor: SummaryWriter,
            proj_eta: ProjectFinishTime, epoch: int):
    avg_reward = log_stats_core(buff, elapsed, monitor, epoch)

    # gather statistics
    q_val_mean = q_np.mean()
    trg_val_mean = trg_np.mean()
    q_var = 1-(q_np - trg_np).var()/trg_np.var()

    # monitor statistics
    monitor.add_scalar('Loss/v_loss', v_loss, epoch)
    monitor.add_scalar('Loss/value_target', trg_val_mean, epoch)
    monitor.add_scalar('Loss/value_dqn', q_val_mean, epoch)
    monitor.add_scalar('Loss/qf_explaned_var', q_var, epoch)

    monitor.add_scalar('Policy/epsilon', epsilon, epoch)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 3) +
          ' reward: {0:.{1}f},'.format(avg_reward, 2) +
          ' v loss: {0:.{1}f},'.format(v_loss, 3) +
          ' q out: {0:.{1}f},'.format(q_val_mean, 2) +
          ' trg out: {0:.{1}f},'.format(trg_val_mean, 2) +
          ' eps: {0:.{1}f}'.format(epsilon, 2))


def log_stats_basic(buff: TransitionBuffer, elapsed: float, monitor: SummaryWriter, proj_eta: ProjectFinishTime,
                    epoch: int):
    avg_reward = log_stats_core(buff, elapsed, monitor, epoch)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 3) +
          ' reward: {0:.{1}f},'.format(avg_reward, 2))


def print_eval(buff: TransitionBuffer, elapsed: float, epoch: int):
    avg_reward = buff.reward_fifo.mean()
    assert buff.b_epi > 0
    epi_len = buff.episode_len_fifo[:buff.b_epi]
    epi_rew = buff.episode_rew_fifo[:buff.b_epi]

    # print results
    print('Evaluation Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 3) +
          ' episode reward: {0:.{1}f} +- {2:.{3}f}s,'.format(epi_rew.mean(), 2, epi_rew.std(), 2) +
          ' episode length: {0:.{1}f} +- {2:.{3}f}s,'.format(epi_len.mean(), 1, epi_len.std(), 1) +
          ' reward: {0:.{1}f},'.format(avg_reward, 2))


def log_stats_core(buff: TransitionBuffer, elapsed: float, monitor: SummaryWriter, epoch: int,
                   eval_mode: bool = False) -> float:
    # gather statistics
    avg_reward = buff.reward_fifo.mean()
    term_mean = buff.terminate_fifo.mean()
    trunc_mean = buff.truncate_fifo.mean()
    epi_len = buff.episode_len_fifo[:buff.b_epi]
    epi_rew = buff.episode_rew_fifo[:buff.b_epi]

    if not eval_mode:
        monitor.add_scalar('State/terminated', term_mean, epoch)
        monitor.add_scalar('State/truncated', trunc_mean, epoch)
        if len(epi_len) > 0:
            monitor.add_scalar('State/episode_length', epi_len.mean(), epoch)
            monitor.add_scalar('State/episode_reward', epi_rew.mean(), epoch)
        monitor.add_scalar('Policy/avg_reward', avg_reward, epoch)
        monitor.add_scalar('Time/elapsed', elapsed, epoch)
        dim_mean_act = buff.action_fifo.mean(axis=tuple(range(buff.action_fifo.ndim - 1)))
        for i in range(dim_mean_act.shape[0]):
            monitor.add_scalar('Act/dim%d' % i, dim_mean_act[i], epoch)
        dim_mean_obs = buff.states_fifo.mean(axis=0)
        dim_std_obs = buff.states_fifo.std(axis=0)
        for i in range(dim_mean_obs.shape[0]):
            monitor.add_scalar('Obs/avg_dim%d' % i, dim_mean_obs[i], epoch)
            monitor.add_scalar('Obs/std_dim%d' % i, dim_std_obs[i], epoch)
    else:
        if len(epi_len) > 0:
            monitor.add_scalar('Eval/episode_length', epi_len.mean(), epoch)
            monitor.add_scalar('Eval/episode_reward', epi_rew.mean(), epoch)
        monitor.add_scalar('Eval/elapsed', elapsed, epoch)

    return avg_reward
