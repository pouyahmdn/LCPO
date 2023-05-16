import pathlib
import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE


# Inspired by https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv
def tabulate_events(run_dir: str):
    assert run_dir.endswith('/')

    tb_log_list = list(pathlib.Path(run_dir).glob('202*/'))
    assert len(tb_log_list) == 1, run_dir
    tb_path = tb_log_list[0]
    assert len(os.listdir(tb_path)) == 1
    summary_iterator = EventAccumulator(os.path.join(tb_path, os.listdir(tb_path)[0]),
                                        size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE).Reload()
    tags = summary_iterator.Tags()['scalars']
    # tags = [tag for tag in tags if tag != 'Adam/PolLR']

    out = {}
    steps = [e.step for e in summary_iterator.Scalars(tags[0])]
    assert min(steps) == 0
    max_steps = max(steps) + 1
    out['_wallclock_min'] = np.ones(max_steps) * np.inf
    out['_wallclock_max'] = np.ones(max_steps) * -np.inf
    for tag in tags:
        out[tag] = np.ones(max_steps) * np.nan
        for event in summary_iterator.Scalars(tag):
            assert np.isnan(out[tag][event.step])
            out['_wallclock_min'][event.step] = min(event.wall_time, out['_wallclock_min'][event.step])
            out['_wallclock_max'][event.step] = max(event.wall_time, out['_wallclock_max'][event.step])
            out[tag][event.step] = event.value

    pd.DataFrame(out, index=steps).to_pickle(f'{run_dir}/tb.pkl')
