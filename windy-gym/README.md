# Locally Constrained Policy Optimization: Windy Pendulum-v1

To reproduce the results in the paper:

0. Install required python packages
1. Run experiments
2. Plot results

---
## 0. Python packages
We use Python (3.8 tested) for all experiments. Install PyTorch according to the website [instructions](https://pytorch.org).
Install the remaining required packages via `pip` (or `conda`):
```
python3 -m pip install -r requirements.txt
```

---
## 1. Run experiments

Ensure you have ~1.5 TiB of free space. Then, run the following command:
```
python3 launch_multi_exp.py --gpu_avail_ind -1 --job_per_gpu 32 --config_file ./run_config_phase1.py --output_dir ./tests/ --free_lim 1500
python3 launch_multi_exp.py --gpu_avail_ind -1 --job_per_gpu 32 --config_file ./run_config_phase2.py --output_dir ./tests/ --free_lim 1500
```
These scripts will run 2680 experiments in 32 parallel streams. The results are saved in `./tests/`. While each run is quick and has low I/O footprint, the scale of experiments demands heavy compute and disk space resources.

If space requirements are an issue, you can:
1. Run the experiments in batches by commenting certain ones in `run_config_phase1.py` or `run_config_phase2.py`.
2. Omit certain logged variables in `agent/monitoring/core_log.py`, but ensure you do not omit 'State/episode_reward'.
3. Not save models by passing `--save_interval 1000000` to all runs in `run_config_phase1.py` or `run_config_phase2.py`.

---
## 2. Plot results

Run the following command:
```
python3 plot_all_figures.py
```
The resulting figures are saved in `./figures/`.