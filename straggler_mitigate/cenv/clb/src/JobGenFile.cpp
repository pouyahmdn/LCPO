//
// Created by Pouya Hamadanian on 5/19/21.
//

#include <iostream>
#include "JobGenFile.h"
#include "utils.h"
#include "cassert"

JobGenFile::JobGenFile(double **size_arrs_orig, double **arrival_arrs_orig, const int *len_arrs_orig, short num_traces,
                       int start_state, double hold_time, int num_servers,
                       unsigned int start_seed, double rate_chosen) {
    this->start_state = start_state;
    this->state = start_state;
    this->state_saved = start_state;
    this->time_elapsed = 0;
    this->time_elapsed_saved = 0;

    this->num_servers = num_servers;
    this->num_traces = num_traces;
    this->rate_chosen = rate_chosen;
    this->hold_time = hold_time;

    this->rng = new std::mt19937(start_seed);
    this->relocate_state_dist = new std::uniform_int_distribution<int>(0, num_traces - 1);
    this->relocate_time_dist = new std::uniform_real_distribution<double>(0, hold_time);
    this->relocate_step_dist = new std::uniform_int_distribution<int> *[num_traces];

    this->size_arrays = new double *[num_traces];
    this->arrival_arrays = new double *[num_traces];
    this->time_arrays = new double *[num_traces];
    this->len_arrays = new int[num_traces];
    this->size_avg = new double[num_traces];
    this->iat_avg = new double[num_traces];
    this->step = new int[num_traces]();
    this->step_saved = new int[num_traces]();
    for (int i = 0; i < num_traces; i++) {
        this->size_arrays[i] = new double[len_arrs_orig[i]];
        this->arrival_arrays[i] = new double[len_arrs_orig[i]];
        this->time_arrays[i] = new double[len_arrs_orig[i]];
        this->len_arrays[i] = len_arrs_orig[i];
        this->relocate_step_dist[i] = new std::uniform_int_distribution<int>(0, len_arrs_orig[i] - 1);
        double prev_time = 0;
        for (int j = 0; j < len_arrs_orig[i]; j++) {
            this->size_arrays[i][j] = size_arrs_orig[i][j];
            this->arrival_arrays[i][j] = arrival_arrs_orig[i][j];
            this->time_arrays[i][j] = prev_time + arrival_arrs_orig[i][j];
            prev_time = this->time_arrays[i][j];
        }
        this->size_avg[i] = average(this->size_arrays[i], this->len_arrays[i]);
        this->iat_avg[i] = average(this->arrival_arrays[i], this->len_arrays[i]);
    }

    this->have_indices = false;
    this -> duplicate_object = false;
}

JobGenFile::JobGenFile(JobGenFile *job_gen) {
    this->start_state = job_gen->start_state;
    this->state = job_gen->state;
    this->state_saved = job_gen->state_saved;
    this->time_elapsed = job_gen->time_elapsed;
    this->time_elapsed_saved = job_gen->time_elapsed_saved;

    this->num_servers = job_gen->num_servers;
    this->num_traces = job_gen->num_traces;
    this->rate_chosen = job_gen->rate_chosen;
    this->hold_time = job_gen->hold_time;

    this->rng = new std::mt19937(*(job_gen->rng));
    this->relocate_state_dist = new std::uniform_int_distribution<int>(0, num_traces - 1);
    this->relocate_time_dist = new std::uniform_real_distribution<double>(0, hold_time);
    this->relocate_step_dist = new std::uniform_int_distribution<int> *[num_traces];
    for (int i = 0; i < num_traces; i++)
        this->relocate_step_dist[i] = new std::uniform_int_distribution<int>(0, job_gen->len_arrays[i] - 1);

    this->size_arrays = job_gen->size_arrays;
    this->arrival_arrays = job_gen->arrival_arrays;
    this->time_arrays = job_gen->time_arrays;
    this->len_arrays = job_gen->len_arrays;
    this->size_avg = job_gen->size_avg;
    this->iat_avg = job_gen->iat_avg;

    // Will this work?
    this->step = new int[num_traces];
    this->step_saved = new int[num_traces];
    for (int i = 0; i < num_traces; i++){
        this->step[i] = job_gen->step[i];
        this->step_saved[i] = job_gen->step_saved[i];
    }

    this->have_indices = job_gen->have_indices;
    if (job_gen->have_indices)
        this->indices = job_gen->indices;
    this -> duplicate_object = true;
}

JobGenFile::~JobGenFile() {
    if (!duplicate_object){
        for (int i = 0; i < num_traces; i++) {
            delete size_arrays[i];
            delete arrival_arrays[i];
            delete time_arrays[i];
            if (have_indices)
                delete indices[i];
        }
        delete size_arrays;
        delete arrival_arrays;
        delete time_arrays;
        if (have_indices)
            delete indices;
        delete len_arrays;
        delete size_avg;
        delete iat_avg;
    }

    delete rng;

    delete step;
    delete step_saved;
    for (int i = 0; i < num_traces; i++) {
        delete relocate_step_dist[i];
    }
    delete relocate_state_dist;
    delete relocate_time_dist;
    delete relocate_step_dist;
}

void JobGenFile::seed(unsigned int start_seed) {
    delete rng;
    this->rng = new std::mt19937(start_seed);
}

void JobGenFile::reset() {
    state = start_state;
    time_elapsed = 0;
    state_saved = start_state;
    time_elapsed_saved = 0;
    for (int i = 0; i < num_traces; i++) {
        step[i] = 0;
        step_saved[i] = 0;
    }
}

double JobGenFile::get_size_avg() {
    return average(size_avg, num_traces);
}

double JobGenFile::get_arrival_avg() {
    double sum = 0;
    for (int i = 0; i < num_traces; i++)
        sum += 1/iat_avg[i];
    return sum/num_traces;
}

int JobGenFile::get_curr_trace() {
    return state;
}

int JobGenFile::get_num_traces() {
    return num_traces;
}

double JobGenFile::get_chosen_rate() {
    return rate_chosen;
}

void JobGenFile::save_state() {
    state_saved = state;
    time_elapsed_saved = time_elapsed;
    for (int i = 0; i < num_traces; i++)
        step_saved[i] = step[i];
}

void JobGenFile::load_state() {
    state = state_saved;
    time_elapsed = time_elapsed_saved;
    for (int i = 0; i < num_traces; i++)
        step[i] = step_saved[i];
}

void JobGenFile::relocate() {
    state = (*relocate_state_dist)(*rng);
    time_elapsed = (*relocate_time_dist)(*rng);
    step[state] = (*relocate_step_dist[state])(*rng);
}

void JobGenFile::report() {
    for (int i = 0; i < num_traces; i++) {
        std::cout << "With state " << i << ": " << std::endl;

        std::cout << "Size has an average of = " << size_avg[i] << " ms and quantiles of:" << std::endl;
        std::cout << "p50 = " << percentile(size_arrays[i], len_arrays[i], 0.5);
        std::cout << ", above p50 average = " << avg_partition_multi_array(&size_arrays[i], &len_arrays[i], 1, 0.5) << std::endl;
        std::cout << "p90 = " << percentile(size_arrays[i], len_arrays[i], 0.9);
        std::cout << ", above p90 average = " << avg_partition_multi_array(&size_arrays[i], &len_arrays[i], 1, 0.1) << std::endl;
        std::cout << "p99 = " << percentile(size_arrays[i], len_arrays[i], 0.99);
        std::cout << ", above p99 average = " << avg_partition_multi_array(&size_arrays[i], &len_arrays[i], 1, 0.01) << std::endl;
        std::cout << "p99.9 = " << percentile(size_arrays[i], len_arrays[i], 0.999);
        std::cout << ", above p99.9 average = " << avg_partition_multi_array(&size_arrays[i], &len_arrays[i], 1, 0.001) << std::endl;

        std::cout << "Arrival has an average of = " << iat_avg[i] << " ms and quantiles of:" << std::endl;
        std::cout << "p50 = " << percentile(arrival_arrays[i], len_arrays[i], 0.5);
        std::cout << ", below p50 average = " << iat_avg[i] * 2 - avg_partition_multi_array(&arrival_arrays[i], &len_arrays[i], 1, 0.5) << std::endl;
        std::cout << "p10 = " << percentile(arrival_arrays[i], len_arrays[i], 0.1);
        std::cout << ", below p10 average = " << iat_avg[i] * 10 - 9 * avg_partition_multi_array(&arrival_arrays[i], &len_arrays[i], 1, 0.9) << std::endl;
        std::cout << "p1 = " << percentile(arrival_arrays[i], len_arrays[i], 0.01);
        std::cout << ", below p1 average = " << iat_avg[i] * 100 - 99 * avg_partition_multi_array(&arrival_arrays[i], &len_arrays[i], 1, 0.99) << std::endl;
        std::cout << "p0.1 = " << percentile(arrival_arrays[i], len_arrays[i], 0.001);
        std::cout << ", below p0.1 average = " << iat_avg[i] * 1000 - 999 * avg_partition_multi_array(&arrival_arrays[i], &len_arrays[i], 1, 0.999) << std::endl;

        std::cout << "Average load = " << size_avg[i] / iat_avg[i] * 1.9 / num_servers / rate_chosen << std::endl;
    }
}

void JobGenFile::load_index_arr(short **indices_orig) {
    this->indices = new short *[num_traces];
    for (int i = 0; i < num_traces; i++) {
        this->indices[i] = new short [len_arrays[i]];
        for (int j = 0; j < len_arrays[i]; j++) {
            this->indices[i][j] = indices_orig[i][j];
        }
    }
    have_indices = true;
}

int JobGenFile::get_curr_index() {
    assert(have_indices);
    return indices[state][step[state]];
}

void JobGenFile::set_curr_trace(int new_state){
    state = new_state;
    std::cout << "In trace mode, state changed to " << state << std::endl;
}

JobProcessSample JobGenFile::gen_job() {
    JobProcessSample ret_sample{
            .size=size_arrays[state][step[state]],
            .inter_arrival_time=arrival_arrays[state][step[state]],
            .trace_origin_index=(unsigned char)state
    };

    step[state] += 1;
    time_elapsed += ret_sample.inter_arrival_time;

    if (step[state] >= len_arrays[state])
        step[state] = 0;

    if (time_elapsed > hold_time and num_traces > 1) {
        state = (state + 1) % num_traces;
        time_elapsed -= hold_time;
        std::cout << "In trace mode, state changed to " << state << std::endl;
    }

    return ret_sample;
}

void JobGenFile::seek(double time_point) {
    step[state] = binary_search_right_side(time_arrays[state], len_arrays[state], time_point);
}

JobGenFile *JobGenFile::copy() {
    return new JobGenFile(this);
}
