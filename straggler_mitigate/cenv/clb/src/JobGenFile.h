//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_JOBGENFILE_H
#define CLB_JOBGENFILE_H

#include <random>
#include "JobGen.h"

class JobGenFile : public JobGen{
private:
    std::mt19937* rng;
    double** size_arrays;
    double** arrival_arrays;
    double** time_arrays;
    short** indices;
    bool have_indices;
    short num_traces;
    int* len_arrays;
    double* iat_avg;
    double* size_avg;

    std::uniform_int_distribution<int>* relocate_state_dist;
    std::uniform_int_distribution<int>** relocate_step_dist;
    std::uniform_real_distribution<double>* relocate_time_dist;

    int start_state;

    int state;
    int state_saved;
    int* step;
    int* step_saved;
    double time_elapsed;
    double time_elapsed_saved;

    double hold_time;
    double rate_chosen;

    bool duplicate_object;

public:
    JobGenFile(double** size_arrs_orig, double** arrival_arrs_orig, const int* len_arrs_orig, short num_traces,
               int start_state, double hold_time, int num_servers, unsigned int start_seed,
               double rate_chosen);
    explicit JobGenFile(JobGenFile* job_gen);
    ~JobGenFile() override;
    void seed(unsigned int start_seed) override;
    void reset() override;
    void report() override;
    double get_size_avg() override;
    double get_arrival_avg() override;
    void load_index_arr(short** indices_orig) override;
    int get_curr_trace() override;
    void set_curr_trace(int new_state) override;
    int get_curr_index() override;
    int get_num_traces() override;
    double get_chosen_rate() override;
    void seek(double time_point);
    void relocate() override;
    JobProcessSample gen_job() override;
    void save_state() override;
    void load_state() override;
    JobGenFile* copy() override;
};


#endif
