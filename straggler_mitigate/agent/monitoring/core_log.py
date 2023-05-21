import numpy as np
from torch.utils.tensorboard import SummaryWriter

from buffer.buffer import TransitionBuffer
from buffer.buffer_disc import DiscontinuousTransitionBuffer
from buffer.buffer_sac import TransitionBuffer as TransitionBufferSAC
from utils.proj_time import ProjectFinishTime


def log_a2c(buff: TransitionBuffer or DiscontinuousTransitionBuffer, ret_np: np.ndarray, v_np: np.ndarray,
            adv_np: np.ndarray, pg_loss: float, v_loss: float, entropy_factor: float, norm_entropy: float,
            log_pi_min: float, tw_ratio: float, elapsed: float, start_wall_time: float, monitor: SummaryWriter,
            proj_eta: ProjectFinishTime, epoch: int, curr_time: float, timeline_len, index_model: int):
    avg_reward, jct_arr, jct_len = log_stats_core(buff, elapsed, start_wall_time, monitor, epoch, curr_time,
                                                  timeline_len, tw_ratio)

    # gather statistics
    ret_mean = ret_np.mean()
    v_net_mean = v_np.mean()
    vf_var = 1-v_loss/v_np.var()
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
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 2) +
          ' reward: {0:.{1}f},'.format(avg_reward.item(), 2) +
          ' model: {0},'.format(index_model) +
          ' p95_delay: {0:.{1}f},'.format(np.mean(np.partition(jct_arr, -jct_len)[-jct_len:]), 2) +
          ' avg_delay: {0:.{1}f},'.format(np.mean(jct_arr), 2) +
          ' v loss: {0:.{1}f},'.format(v_loss, 3) +
          ' v out: {0:.{1}f},'.format(v_net_mean, 2) +
          ' ret out: {0:.{1}f},'.format(ret_mean, 2) +
          ' entropy: {0:.{1}f}'.format(norm_entropy, 2))


def log_stats_sac(buff: TransitionBufferSAC, q_target: float, q_local_1: float, q_local_2: float, pg_loss: float,
                  q_loss_1: float, q_loss_2: float, entropy_factor: float, norm_entropy: float, tw_ratio: float,
                  elapsed: float, start_wall_time: float, monitor: SummaryWriter,
                  proj_eta: ProjectFinishTime, epoch: int, curr_time: float, timeline_len: int):

    # gather statistics
    avg_reward = buff.reward_fifo[:buff.episode_len].mean()
    mask_mean = 1-buff.states_fifo[:buff.episode_len, -1].mean()
    done_mean = buff.dones_fifo[:buff.episode_len].mean()


    # monitor statistics
    monitor.add_scalar('Loss/mask_mean', mask_mean, epoch)

    monitor.add_scalar('State/done', done_mean, epoch)
    monitor.add_scalar('State/train_wheels_engage', tw_ratio, epoch)

    monitor.add_scalar('Reward/avg_reward', avg_reward, epoch)

    monitor.add_scalar('Time/elapsed', elapsed, epoch)
    monitor.add_scalar('Time/total_elapsed', (curr_time - start_wall_time) / 1000 / 3600, epoch)
    monitor.add_scalar('Time/timeline_len', timeline_len, epoch)

    monitor.add_scalar('Policy/max_timeout', buff.action_timeout_fifo[:buff.episode_len].max(), epoch)
    monitor.add_scalar('Policy/min_timeout', buff.action_timeout_fifo[:buff.episode_len].min(), epoch)
    monitor.add_scalar('Policy/avg_timeout', buff.action_timeout_fifo[:buff.episode_len].mean(), epoch)

    monitor.add_scalar('Load/jobs_per_second', buff.workload_fifo.mean(), epoch)

    dim_mean_obs = buff.states_fifo[:buff.episode_len].mean(axis=tuple(range(buff.states_fifo.ndim - 1)))
    for i in range(dim_mean_obs.shape[0]):
        monitor.add_scalar('Obs/dim%d' % i, dim_mean_obs[i], epoch)

    # monitor statistics
    monitor.add_scalar('Loss/pg_loss', pg_loss, epoch)
    monitor.add_scalar('Loss/v_loss_1', q_loss_1, epoch)
    monitor.add_scalar('Loss/v_loss_2', q_loss_2, epoch)
    monitor.add_scalar('Loss/q_target', q_target, epoch)
    monitor.add_scalar('Loss/q_local_1', q_local_1, epoch)
    monitor.add_scalar('Loss/q_local_2', q_local_2, epoch)

    monitor.add_scalar('Policy/entropy_factor', entropy_factor, epoch)
    monitor.add_scalar('Policy/norm_entropy', norm_entropy, epoch)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 2) +
          ' reward: {0:.{1}f},'.format(avg_reward.item(), 2) +
          ' pg loss: {0:.{1}f},'.format(pg_loss, 3) +
          ' q loss: {0:.{2}f} {1:.{2}f},'.format(q_loss_1, q_loss_2, 3) +
          ' q out: {0:.{2}f} {1:.{2}f},'.format(q_local_1, q_local_2, 2) +
          ' target out: {0:.{1}f}'.format(q_target, 2))


def log_dqn(buff: TransitionBufferSAC, q_np: np.ndarray, trg_np: np.ndarray, v_loss: float, epsilon: float,
            tw_ratio: float, elapsed: float, start_wall_time: float, monitor: SummaryWriter,
            proj_eta: ProjectFinishTime, epoch: int, curr_time: float, timeline_len: int):
    avg_reward, jct_arr, jct_len = log_stats_core(buff, elapsed, start_wall_time, monitor, epoch, curr_time,
                                                  timeline_len, tw_ratio)

    # gather statistics
    q_val_mean = q_np.mean()
    trg_val_mean = trg_np.mean()

    # monitor statistics
    monitor.add_scalar('Loss/v_loss', v_loss, epoch)
    monitor.add_scalar('Loss/value_target', trg_val_mean, epoch)
    monitor.add_scalar('Loss/value_dqn', q_val_mean, epoch)

    monitor.add_scalar('Policy/epsilon', epsilon, epoch)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 2) +
          ' reward: {0:.{1}f},'.format(avg_reward.item(), 2) +
          ' p95_delay: {0:.{1}f},'.format(np.mean(np.partition(jct_arr, -jct_len)[-jct_len:]), 2) +
          ' avg_delay: {0:.{1}f},'.format(np.mean(jct_arr), 2) +
          ' v loss: {0:.{1}f},'.format(v_loss, 3) +
          ' q out: {0:.{1}f},'.format(q_val_mean, 2) +
          ' trg out: {0:.{1}f},'.format(trg_val_mean, 2) +
          ' eps: {0:.{1}f}'.format(epsilon, 2))


def log_stats_basic(buff: TransitionBuffer, elapsed: float, start_wall_time: float, monitor: SummaryWriter,
                    proj_eta: ProjectFinishTime, epoch: int, curr_time: float, timeline_len: int, tw_ratio: float):
    avg_reward, jct_arr, jct_len = log_stats_core(buff, elapsed, start_wall_time, monitor, epoch, curr_time,
                                                  timeline_len, tw_ratio)

    # print results
    proj_eta.update_progress(epoch)
    print('Epoch: {},'.format(epoch) +
          ' elapsed: {0:.{1}f}s,'.format(elapsed, 2) +
          ' reward: {0:.{1}f},'.format(avg_reward.item(), 2) +
          ' p95_delay: {0:.{1}f},'.format(np.mean(np.partition(jct_arr, -jct_len)[-jct_len:]), 2) +
          ' avg_delay: {0:.{1}f},'.format(np.mean(jct_arr), 2))


def log_stats_core(buff: TransitionBuffer, elapsed: float, start_wall_time: float, monitor: SummaryWriter,
                   epoch: int, curr_time: float, timeline_len: int, tw_ratio: float):
    # gather statistics
    avg_reward = buff.reward_fifo[:buff.episode_len].mean()
    avg_queue_size = buff.state_queue_size_fifo[:buff.episode_len].mean()
    max_queue_size = buff.state_queue_size_fifo[:buff.episode_len].max()
    mask_mean = 1-buff.states_fifo[:buff.episode_len, -1].mean()
    done_mean = buff.dones_fifo[:buff.episode_len].mean()

    # get parameter scale
    t_s_min, t_s_avg, t_s_max = buff.get_server_load()
    jct_len = int(len(buff.get_job_completion_times()) * 0.1)
    jct_arr = np.array(buff.get_job_completion_times())

    # monitor statistics
    monitor.add_scalar('Loss/mask_mean', mask_mean, epoch)

    monitor.add_scalar('State/avg_queue_size', avg_queue_size, epoch)
    monitor.add_scalar('State/max_queue_size', max_queue_size, epoch)
    monitor.add_scalar('State/done', done_mean, epoch)
    monitor.add_scalar('State/train_wheels_engage', tw_ratio, epoch)

    monitor.add_scalar('Reward/avg_reward', avg_reward, epoch)
    monitor.add_scalar('Reward/p95_delay', np.mean(np.partition(jct_arr, -jct_len)[-jct_len:]), epoch)
    monitor.add_scalar('Reward/avg_delay', np.mean(jct_arr), epoch)

    monitor.add_scalar('Time/elapsed', elapsed, epoch)
    monitor.add_scalar('Time/total_elapsed', (curr_time - start_wall_time) / 1000 / 3600, epoch)
    monitor.add_scalar('Time/timeline_len', timeline_len, epoch)

    monitor.add_scalar('Policy/max_timeout', buff.action_timeout_fifo[:buff.episode_len].max(), epoch)
    monitor.add_scalar('Policy/min_timeout', buff.action_timeout_fifo[:buff.episode_len].min(), epoch)
    monitor.add_scalar('Policy/avg_timeout', buff.action_timeout_fifo[:buff.episode_len].mean(), epoch)

    monitor.add_scalar('Load/avg_load', t_s_avg, epoch)
    monitor.add_scalar('Load/min_load', t_s_min, epoch)
    monitor.add_scalar('Load/max_load', t_s_max, epoch)
    monitor.add_scalar('Load/jobs_per_second', buff.workload_fifo.mean(), epoch)

    dim_mean_obs = buff.states_fifo[:buff.episode_len].mean(axis=tuple(range(buff.states_fifo.ndim - 1)))
    for i in range(dim_mean_obs.shape[0]):
        monitor.add_scalar('Obs/dim%d' % i, dim_mean_obs[i], epoch)

    monitor.add_histogram('Action/actions', buff.action_timeout_fifo[:buff.episode_len], epoch)
    monitor.add_histogram('Time/JCT', jct_arr)

    return avg_reward, jct_arr, jct_len
