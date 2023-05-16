import itertools
import os
import shutil
import signal
import subprocess
import time
import argparse
from termcolor import colored

parser = argparse.ArgumentParser(description='Auto experiment launch')
parser.add_argument('--gpu_avail_ind', type=int, nargs='+', required=True, help='Index of available GPUs, -1 for CPU')
parser.add_argument('--job_per_gpu', type=int, nargs='+', required=True, help='Number of available GPUs')
parser.add_argument('--config_file', type=str, required=True, help='Run configuration file')
parser.add_argument('--output_dir', type=str, required=True, help='Run output dir')
parser.add_argument('--print_cmd_only', action='store_true', help='Print commands and exit')
parser.add_argument('--tail', action='store_true', help='Tail experiments')
parser.add_argument('--only_running', action='store_true', help='Tail experiments')
parser.add_argument('--free_lim', type=float, default=-1, help='Minimum hard drive space to start')


args = parser.parse_args()

assert len(args.job_per_gpu) == 1 or len(args.job_per_gpu) == len(args.gpu_avail_ind)
assert all(gpu == -1 or gpu >= 0 for gpu in args.gpu_avail_ind)
if len(args.job_per_gpu) == 1:
    args.job_per_gpu = [args.job_per_gpu[0] for _ in range(len(args.gpu_avail_ind))]


def print_distinct(str_print):
    print(colored(str_print, on_color='on_cyan', attrs=['bold']))


def print_cmd(str_print):
    print(colored(str_print, 'green', attrs=['bold']))


def main():
    if args.output_dir[-1] != '/':
        args.output_dir += '/'
    with open(args.config_file, 'r') as f:
        config_init = eval(f.read())
    runs = []
    for run_dict in config_init:
        assert isinstance(run_dict['common'], str)
        assert isinstance(run_dict['name'], str)
        assert isinstance(run_dict['iterant'], list)
        common_cmd = run_dict['common']
        iter_name = []
        iter_list = []
        iter_rep = []
        for iter_dict in run_dict['iterant']:
            assert isinstance(iter_dict, dict)
            assert isinstance(iter_dict['val'], list)
            rep = 'rep' if 'rep' in iter_dict else 'val'
            if isinstance(iter_dict['arg'], str):
                iter_name.append((iter_dict['arg'], ))
                iter_list.append([(val, ) for val in iter_dict['val']])
                iter_rep.append([(val, ) for val in iter_dict[rep]])
                for next_stuff in iter_dict['val']:
                    assert isinstance(next_stuff, str)
                for next_stuff in iter_dict[rep]:
                    assert isinstance(next_stuff, str)
            else:
                assert isinstance(iter_dict['arg'], tuple)
                iter_name.append(iter_dict['arg'])
                for something in iter_dict['arg']:
                    assert isinstance(something, str)
                iter_list.append(iter_dict['val'])
                for something in iter_dict['val']:
                    assert isinstance(something, tuple)
                    for next_stuff in something:
                        assert isinstance(next_stuff, str)
                if isinstance(iter_dict[rep][0], str):
                    iter_rep.append([(val,) for val in iter_dict[rep]])
                    for next_stuff in iter_dict[rep]:
                        assert isinstance(next_stuff, str)
                else:
                    iter_rep.append(iter_dict[rep])
                    for something in iter_dict[rep]:
                        assert isinstance(something, tuple)
                        for next_stuff in something:
                            assert isinstance(next_stuff, str)

        for arg_comb, rep_comb in zip(itertools.product(*iter_list), itertools.product(*iter_rep)):
            result_directory = args.output_dir + run_dict['name']
            name_run = run_dict['name']
            run_command = common_cmd
            for index_group, arg_group in enumerate(arg_comb):
                rep_group = rep_comb[index_group]
                single_rep = len(rep_group) != len(arg_group)
                for index_arg, arg_name in enumerate(iter_name[index_group]):
                    if arg_group[index_arg][:3] == '###':
                        if arg_group[index_arg][3:] == 'YES':
                            run_command = f'{run_command} --{iter_name[index_group][index_arg]}'
                    else:
                        run_command = f'{run_command} --{iter_name[index_group][index_arg]} {arg_group[index_arg]}'
                    if not single_rep:
                        if arg_group[index_arg][:3] == '###':
                            state_store = 'with' if arg_group[index_arg] == '###YES' else 'no'
                            result_directory = f'{result_directory}_{state_store}_{iter_name[index_group][index_arg]}'
                            name_run = f'{name_run}_{state_store}_{iter_name[index_group][index_arg]}'
                        else:
                            result_directory = f'{result_directory}_{iter_name[index_group][index_arg]}' \
                                               f'{rep_group[index_arg]}'
                            name_run = f'{name_run}_{iter_name[index_group][index_arg]}{rep_group[index_arg]}'
                if single_rep:
                    result_directory = f'{result_directory}_{rep_group[0]}'
                    name_run = f'{name_run}_{rep_group[0]}'
            result_directory = result_directory + '/'
            run_command = f'{run_command} --result_folder {result_directory}'
            runs.append({'cmd': run_command, 'dir': result_directory, 'name': name_run})

    print_distinct('Run list created, %d runs!!!' % len(runs))
    num_total_jobs = len(runs)

    if args.print_cmd_only:
        for run in runs:
            print(run['cmd'])
        exit(0)

    if args.tail:
        for run in runs:
            if not os.path.exists(run['dir']):
                continue
            if args.only_running and not (os.path.exists(f"{run['dir']}/run_in_play") or
                                          os.path.exists(f"{run['dir']}/log_in_play")):
                continue
            print_distinct(f'For experiment {run["name"]}:')
            subprocess.Popen(['tail', '-n', '10', run['dir']+'train.log']).wait()
            print()
        exit(0)

    num_complete_jobs = handle_jobs(runs)

    print_distinct('Exiting with %d/%d jobs completed right...' % (num_complete_jobs, num_total_jobs))


def handle_jobs(list_runs):
    correctly_done_jobs = 0
    active_jobs = []
    signal.signal(signal.SIGINT, signal.default_int_handler)

    gpu_map = args.gpu_avail_ind
    gpu_reserved = [i for i in range(len(args.gpu_avail_ind))]
    gpu_available = [args.job_per_gpu[i] for i in range(len(args.gpu_avail_ind))]
    root_dir = '/' + (os.path.abspath(args.output_dir).split("/")[1]) + '/'
    print(f'Root directory is {root_dir}')

    try:
        while len(active_jobs) > 0 or len(list_runs) > 0:

            total, used, free = shutil.disk_usage(root_dir)
            
            for i in range(len(gpu_available)):
                if gpu_available[i] > 0 and len(list_runs) > 0 and (args.free_lim < 0 or free > args.free_lim * 2**30):
                    new_run = list_runs.pop(0)
                    job = {'index': i, 'config': new_run}
                    device = 'cpu' if gpu_map[i] == -1 else 'cuda:%d' % gpu_map[i]
                    cmd_final = new_run['cmd'] + ' --device %s' % device
                    print_distinct(f'{free // 2**30} GiB free on /data/')
                    print_cmd(cmd_final)
                    gpu_available[i] -= 1
                    os.makedirs(new_run['dir'], exist_ok=True)
                    job['file'] = open("%strain.log" % new_run['dir'], "w")
                    job['proc'] = subprocess.Popen(cmd_final, shell=True, preexec_fn=os.setsid,
                                                   stdout=job['file'], stderr=job['file'])
                    active_jobs.append(job)

            to_remove = []
            for job in active_jobs:
                job_idle = not (job['proc'].poll() is None)
                if job_idle:
                    print_distinct("JOB OVER, ANY POSSIBLE ERRORS BELOW:")
                    subprocess.Popen(
                        "grep -Hn --color=auto  -E -i -- 'Traceback|error|, line|Removing|Kill|failed|Aborted|dumped|Terminated' "
                        "%s*.log" %
                        job['config']['dir'], shell=True, preexec_fn=os.setsid).wait()
                    to_remove.append(job)
                    correctly_done_jobs += 1
                    gpu_available[job['index']] += 1
                    job['file'].flush()
                    job['file'].close()
            for job in to_remove:
                active_jobs.remove(job)

            for job in active_jobs:
                job['file'].flush()

            if os.path.exists(args.output_dir + 'kill.grace'):
                print_distinct('Received graceful kill order')
                while len(list_runs) > 0:
                    new_run = list_runs.pop(0)
                    print_distinct(f"Flushing job {new_run['name']}")
                os.remove(args.output_dir + 'kill.grace')

            to_remove_gpu = []
            for gpu in gpu_reserved:
                if os.path.exists(args.output_dir + f'free.gpu.{args.gpu_avail_ind[gpu]}'):
                    print_distinct(f'Received free order for gpu {args.gpu_avail_ind[gpu]}')
                    gpu_available[gpu] -= args.job_per_gpu[gpu]
                    assert gpu_available[gpu] <= 0
                    to_remove_gpu.append(gpu)
                    os.remove(args.output_dir + f'free.gpu.{args.gpu_avail_ind[gpu]}')
            for gpu in to_remove_gpu:
                gpu_reserved.remove(gpu)

            if os.path.exists(args.output_dir + 'pause'):
                print_distinct('Received graceful pause order')
                for job in active_jobs:
                    if job['proc'].poll() is None:
                        os.kill(job['proc'].pid, signal.SIGSTOP)
                        print_distinct(f"Pausing job {job['config']['name']}")
                os.remove(args.output_dir + 'pause')

            if os.path.exists(args.output_dir + 'resume'):
                print_distinct('Received graceful resume order')
                for job in active_jobs:
                    if job['proc'].poll() is None:
                        os.kill(job['proc'].pid, signal.SIGCONT)
                        print_distinct(f"Resuming job {job['config']['name']}")
                os.remove(args.output_dir + 'resume')

            time.sleep(0.2)

    except KeyboardInterrupt:
        print_distinct("CAPTURED KILL SIGNAL")
        for job in active_jobs:
            if job['proc'].poll() is None:
                os.killpg(os.getpgid(job['proc'].pid), signal.SIGKILL)

    return correctly_done_jobs


if __name__ == "__main__":
    main()
