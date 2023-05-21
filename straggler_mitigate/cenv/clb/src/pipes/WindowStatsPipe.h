//
// Created by Pouya Hamadanian on 8/9/22.
//

#ifndef CLB_WINDOWSTATSPIPE_H
#define CLB_WINDOWSTATSPIPE_H

#include "pipe.h"
#include <vector>

template <class T> struct SingleStat{
    T average;
    T minimum;
    T maximum;
    T median;
    T per95;
    T per97;
    T per99;
};

struct WindowStat{
    SingleStat<float> arrival;
    SingleStat<float> delay;
    SingleStat<float> proc_first;
    SingleStat<float> proc;
    SingleStat<float> size;
    SingleStat<float> delay_p1;
    SingleStat<float> delay_proc;
    SingleStat<float> delay_size;
    SingleStat<float> qdelay;
    float inflation_avg;
    float model_avg;
    float tw_avg;
    float interval_duration;
    size_t len;
    unsigned char trace;
};

class WindowStatsPipe: public Pipe {
public:
    WindowStatsPipe() = default;
    ~WindowStatsPipe() override = default;

    void flush() override;
    void enqueue(void* entry) override;
    void extend(std::vector<void*>& arr_entry) override;
};

float quick_percentile(std::vector<float> sorted_arr, float p);

SingleStat<float> get_single_stat(std::vector<float> arr_data, bool sort);

#endif //CLB_WINDOWSTATSPIPE_H
