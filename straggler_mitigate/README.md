# Locally Constrained Policy Optimization: Straggler Mitigation

To reproduce the results in the paper:

1. Compile the environment
2. Run experiments
3. Plot results


## 1. Compile the environment

With Cython installed, compile the environment:
```bash
cd cenv/clb/
make all
cd ../../
```
This script will compile the C++ code for the straggler mitigation environment and bind it as a python library. Take care of the logs and make sure there were no errors (afew warnings are expected).



## 2. Run experiments

Ensure you have ~103 GiB of free space. Then, run the following command:
```bash
python3 launch_multi_exp.py --gpu_avail_ind -1 --job_per_gpu 32 --config_file ./run_config.py --output_dir ./tests/ --free_lim 103
```
This script will run 280 experiments in 32 parallel streams. The results are saved in `./tests/`.


## 3. Plot results

Run the following command:
```bash
python3 plot_all_figures.py
```
The resulting figures are saved in `./figures/`.


### Note about traces

As of this moment, the real workload traces used in the paper are not cleared for public release. We have more detail on this and generating custom traces [here](traces/README.md).