# Locally Constrained Policy Optimization: Discrete Pendulum-v1

To reproduce the results in the paper:

1. Run experiments
2. Print results

## 1. Run experiments

Ensure you have ~2 GiB of free space. Then, run the following command:
```bash
python3 launch_multi_exp.py --gpu_avail_ind -1 --job_per_gpu 50 --config_file ./run_config.py --output_dir ./tests/ --free_lim 2
```
This script will run 50 experiments in parallel. The results are saved in `./tests/`.

## 2. Print results

Run the following command:
```bash
python3 plot_all_figures.py
```
The results for Table 4 are printed out.