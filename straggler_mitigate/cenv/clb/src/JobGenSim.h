//
// Created by Pouya Hamadanian on 5/24/21.
//

#ifndef CLB_JOBGENSIM_H
#define CLB_JOBGENSIM_H

#include "JobGen.h"
#include "dists/distribution.h"

class JobGenSim: public JobGen {
private:
    double BOUND_LOW_SIZE = 3;
    double BOUND_HIGH_SIZE = 500;
    double BOUND_LOW_IAT = 1;
    double BOUND_HIGH_IAT = 250;

    std::mt19937* rng;
    int state;
    double time_elapsed;
    double hold_time;
    int num_states;
    double* iat_avg;
    double* size_avg;
    dist_param* dist_param_size;
    dist_param* dist_param_iat;
    distribution** dists_size;
    distribution** dists_iat;
    std::uniform_int_distribution<int>* relocate_state_dist;
    std::uniform_real_distribution<double>* relocate_time_dist;
    int state_saved;
    double time_elapsed_saved;
    int start_state;

    bool duplicate_object;

public:
    JobGenSim(int start_state, double hold_time, int num_states, dist_param* dist_psize, dist_param* dist_piat, int num_servers,
              unsigned int start_seed);
    explicit JobGenSim(JobGenSim* job_gen);
    ~JobGenSim() override;
    void seed(unsigned int start_seed) override;
    void reset() override;
    void report() override;
    double get_size_avg() override;
    double get_arrival_avg() override;
    void load_index_arr(short** indices) override;
    int get_curr_trace() override;
    void set_curr_trace(int new_state) override;
    int get_num_traces() override;
    int get_curr_index() override;
    double get_chosen_rate() override;
    void relocate() override;
    JobProcessSample gen_job() override;
    void save_state() override;
    void load_state() override;
    JobGenSim* copy() override;
};


#endif
