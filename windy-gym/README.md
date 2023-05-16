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

Ensure you have ~80 GiB of free space. Then, run the following command:
```
python3 launch_multi_exp.py --gpu_avail_ind -1 --job_per_gpu 32 --config_file ./run_config.py --output_dir ./tests/ --free_lim 80
```
This script will run 720 experiments in 32 parallel streams. The results are saved in `./tests/`.

---
## 2. Plot results

Run the following command:
```
python3 plot_all_figures.py
```
The resulting figures are saved in `./figures/`.