'''
Linear projection for the finish time
Timer starts at initialization
Update the projector at each iteration
'''

import time
import numpy as np
from datetime import datetime


class ProjectFinishTime(object):
    def __init__(self, total_steps, init_step=0,
                 print_info=True, same_line=False):
        self.total_steps = total_steps
        self.init_step = init_step
        self.print_info = print_info
        if same_line:
            self.end = '\r'
        else:
            self.end = '\n'
        self.reset_timer()

    def reset_timer(self):
        self.init_time = time.time()

    def update_progress(self, step):
        curr_time = time.time()
        
        done_portion = (step - self.init_step) / \
                       (self.total_steps - self.init_step)
        done_time = curr_time - self.init_time

        if done_portion == 0:
            # too short to project eta
            return

        # project time
        undone_time = done_time * \
            (1 - done_portion) / done_portion

        ud_hr = int(np.floor(undone_time / 3600))
        ud_min = int(np.floor(
            (undone_time - ud_hr * 3600) / 60))
        ud_sec = undone_time % 60

        # estimated finish time
        finish_time = curr_time + undone_time
        finish_time = datetime.fromtimestamp(
            finish_time).strftime("%Y-%m-%d %I:%M:%S")

        if self.print_info:
            print('{} out of {} steps, '.format(
                  step, self.total_steps) +
                  'time left: {}h {}m {}s, '.format(
                  ud_hr, ud_min, round(ud_sec, 2)) +
                  'eta: {}'.format(
                  finish_time), end=self.end)
