# Locally Constrained Policy Optimization: Grid World Toy Example

To reproduce the results in the paper:

1. Run experiments
2. Plot results

## 1. Run experiments

Ensure you have ~1 GiB of free space. Then, run the following command:
```bash
python3 launch_multi_exp.py --gpu_avail_ind -1 --job_per_gpu 12 --config_file ./run_config.py --output_dir ./tests/ --free_lim 1
```
This script will run all experiments in parallel. The results are saved in `./tests/`.

## 2. Plot results

Run the following command:
```bash
python3 plot_all_figures.py
```
The resulting figures are saved in `./figures/`.