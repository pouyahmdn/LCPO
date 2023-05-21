//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_JOBGEN_H
#define CLB_JOBGEN_H

#include <random>

struct JobProcessSample{
    double size;
    double inter_arrival_time;
    unsigned char trace_origin_index;
};

class JobGen {
protected:
    int num_servers;

public:

    virtual ~JobGen() = default;
    virtual void seed(unsigned int start_seed) = 0;
    virtual void reset() = 0;
    virtual void report() = 0;
    virtual double get_size_avg() = 0;
    virtual double get_arrival_avg() = 0;
    virtual void load_index_arr(short** indices) = 0;
    virtual int get_curr_trace() = 0;
    virtual void set_curr_trace(int new_state) = 0;
    virtual int get_curr_index() = 0;
    virtual int get_num_traces() = 0;
    virtual double get_chosen_rate() = 0;
    virtual void relocate() = 0;
    virtual JobProcessSample gen_job() = 0;
    virtual void save_state() = 0;
    virtual void load_state() = 0;
    virtual JobGen* copy() = 0;
};


#endif
