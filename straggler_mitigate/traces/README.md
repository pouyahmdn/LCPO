# Traces

The traces used in the experiments in the LCPO paper are not cleared for public release yet. We will add them here when they are.
We encourage researchers interested in using this environment to contact the authors for access.

## Custom traces
You could also generate your own traces, and change the trace loading path at `cenv/load_balance.py` to load your custom generated traces.
Each `trace` is a numpy array of shape `(N, 2)`, where `N` is the number of jobs in the trace.
For index `i`, `trace[i, 0]` denotes the processing time of that job, and `trace[i, 1]` denotes the interarrival time of job at index `i` (time difference between arrival of job `i` and `i-1`).
A nice starting point for synthetic trace generation is a pareto distribution for processing time and an exponential distribution for inter-arrival time.