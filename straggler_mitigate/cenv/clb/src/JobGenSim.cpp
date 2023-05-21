//
// Created by Pouya Hamadanian on 5/24/21.
//

#include <iostream>
#include "JobGenSim.h"
#include "dists/pareto_distribution.h"
#include "dists/exponential_distribution.h"
#include "dists/normal_dist.h"
#include "dists/static_dist.h"
#include "utils.h"
#include "cassert"

JobGenSim::JobGenSim(int start_state, double hold_time, int num_states, dist_param *dist_psize, dist_param *dist_piat,
                     int num_servers, unsigned int start_seed) {
    this->start_state = start_state;
    this->state = start_state;
    this->state_saved = start_state;
    this->hold_time = hold_time;
    this->time_elapsed = 0;
    this->time_elapsed_saved = 0;
    this->num_servers = num_servers;
    this->num_states = num_states;
    this->rng = new std::mt19937(start_seed);
    this->relocate_state_dist = new std::uniform_int_distribution<int>(0, num_states - 1);
    this->relocate_time_dist = new std::uniform_real_distribution<double>(0, hold_time);
    this->dists_size = new distribution *[num_states];
    this->dists_iat = new distribution *[num_states];
    this->size_avg = new double[num_states];
    this->iat_avg = new double[num_states];
    this->dist_param_iat = new dist_param[num_states];
    this->dist_param_size = new dist_param[num_states];
    for (int i = 0; i < num_states; i++) {
        this->dist_param_size[i] = dist_psize[i];
        this->dist_param_iat[i] = dist_piat[i];
        switch (dist_psize[i].type) {
            case pareto:
                assert(dist_psize[i].pareto_shape > 1);
                this->dists_size[i] = new pareto_dist(dist_psize[i].pareto_shape, dist_psize[i].pareto_scale);
                break;
            case normal:
                this->dists_size[i] = new normal_dist(dist_psize[i].normal_mu, dist_psize[i].normal_sigma);
                break;
            case exponential:
                this->dists_size[i] = new exponential_dist(dist_psize[i].exponential_rev_lambda);
                break;
            case stat_ret:
                this->dists_size[i] = new static_dist(dist_psize[i].stat_ret_val);
                this->size_avg[i] = dist_psize[i].stat_ret_val;
                break;
        }
        this->size_avg[i] = dists_size[i]->average(BOUND_LOW_SIZE, BOUND_HIGH_SIZE);
        switch (dist_piat[i].type) {
            case pareto:
                assert(dist_piat[i].pareto_shape > 1);
                this->dists_iat[i] = new pareto_dist(dist_piat[i].pareto_shape, dist_piat[i].pareto_scale);
                break;
            case normal:
                this->dists_iat[i] = new normal_dist(dist_piat[i].normal_mu, dist_piat[i].normal_sigma);
                break;
            case exponential:
                this->dists_iat[i] = new exponential_dist(dist_piat[i].exponential_rev_lambda);
                break;
            case stat_ret:
                this->dists_iat[i] = new static_dist(dist_piat[i].stat_ret_val);
                break;
        }
        this->iat_avg[i] = dists_iat[i]->average(BOUND_LOW_IAT, BOUND_HIGH_IAT);
    }
    this -> duplicate_object = false;
}

JobGenSim::JobGenSim(JobGenSim *job_gen) {
    this->start_state = job_gen->start_state;
    this->state = job_gen->state;
    this->state_saved = job_gen->state_saved;
    this->hold_time = job_gen->hold_time;
    this->time_elapsed = job_gen->time_elapsed;
    this->time_elapsed_saved = job_gen->time_elapsed_saved;
    this->num_servers = num_servers;
    this->num_states = job_gen->num_states;
    this->rng = new std::mt19937(*(job_gen->rng));
    this->relocate_state_dist = new std::uniform_int_distribution<int>(0, num_states - 1);
    this->relocate_time_dist = new std::uniform_real_distribution<double>(0, hold_time);
    this->dists_size = new distribution *[num_states];
    this->dists_iat = new distribution *[num_states];
    this->dist_param_iat = job_gen->dist_param_iat;
    this->dist_param_size = job_gen->dist_param_size;
    for (int i = 0; i < num_states; i++) {
        this->dists_size[i] = job_gen->dists_size[i]->copy();
        this->dists_iat[i] = job_gen->dists_iat[i]->copy();
    }
    this->size_avg = job_gen->size_avg;
    this->iat_avg = job_gen->iat_avg;
    this -> duplicate_object = true;
}

void JobGenSim::seed(unsigned int start_seed) {
    delete rng;
    this->rng = new std::mt19937(start_seed);
}

void JobGenSim::reset() {
    state = start_state;
    time_elapsed = 0;
    state_saved = start_state;
    time_elapsed_saved = 0;
}

void JobGenSim::save_state() {
    state_saved = state;
    time_elapsed_saved = time_elapsed;
}

void JobGenSim::load_state() {
    state = state_saved;
    time_elapsed = time_elapsed_saved;
}

void JobGenSim::load_index_arr(short** indices) {}

int JobGenSim::get_curr_trace() {
    return state;
}

int JobGenSim::get_num_traces() {
    return num_states;
}

double JobGenSim::get_size_avg() {
    return average(size_avg, num_states);
}

double JobGenSim::get_arrival_avg() {
    double sum = 0;
    for (int i = 0; i < num_states; i++)
        sum += 1/iat_avg[i];
    return sum/num_states;
}

void JobGenSim::relocate() {
    state = (*relocate_state_dist)(*rng);
    time_elapsed = (*relocate_time_dist)(*rng);
}

void JobGenSim::set_curr_trace(int new_state){
    state = new_state;
    std::cout << "In simulation mode, state changed to " << state << std::endl;
}


JobProcessSample JobGenSim::gen_job() {
    JobProcessSample ret_sample{
            .size=dists_size[state]->generate_bounded(rng, BOUND_LOW_SIZE, BOUND_HIGH_SIZE),
            .inter_arrival_time=dists_iat[state]->generate_bounded(rng, BOUND_LOW_IAT, BOUND_HIGH_IAT),
            .trace_origin_index=(unsigned char)state
    };

    time_elapsed += ret_sample.inter_arrival_time;
    if (time_elapsed > hold_time and num_states > 1) {
        state = (state + 1) % num_states;
        time_elapsed -= hold_time;
        std::cout << "In simulation mode, state changed to " << state << std::endl;
    }
    return ret_sample;
}

void JobGenSim::report() {
    std::cout << "Bounding size to [" << BOUND_LOW_SIZE << ", " << BOUND_HIGH_SIZE << "] ";
    std::cout << "and iat to [" << BOUND_LOW_IAT << ", " << BOUND_HIGH_IAT << "]" << std::endl;
    for (int i = 0; i < num_states; i++) {
        std::cout << "With state " << i << ": ";
        switch (dist_param_size[i].type) {
            case pareto:
                std::cout << "Size is pareto with shape " << dist_param_size[i].pareto_shape << " and scale "
                          << dist_param_size[i].pareto_scale;
                break;
            case normal:
                std::cout << "Size is normal(" << dist_param_size[i].normal_mu << ", "
                          << dist_param_size[i].normal_sigma << ")";
                break;
            case exponential:
                std::cout << "Size is exponential(" << dist_param_size[i].exponential_rev_lambda << ")";
                break;
            case stat_ret:
                std::cout << "Size is static=" << dist_param_size[i].stat_ret_val;
                break;
        }
        std::cout << " with average=" << size_avg[i] << " ms and ";
        switch (dist_param_iat[i].type) {
            case pareto:
                std::cout << "iat is pareto with shape " << dist_param_iat[i].pareto_shape << " and scale "
                          << dist_param_iat[i].pareto_scale;
                break;
            case normal:
                std::cout << "iat is normal(" << dist_param_iat[i].normal_mu << ", " << dist_param_iat[i].normal_sigma
                          << ")";
                break;
            case exponential:
                std::cout << "iat is exponential(" << dist_param_iat[i].exponential_rev_lambda << ")";
                break;
            case stat_ret:
                std::cout << "iat is static=" << dist_param_iat[i].stat_ret_val;
                break;
        }
        std::cout << " with average=" << iat_avg[i] << " ms, load=" << size_avg[i] / iat_avg[i] * 1.9 / num_servers
                  << std::endl;
    }
}

double JobGenSim::get_chosen_rate() {
    return 1;
}

JobGenSim::~JobGenSim() {
    if (!duplicate_object){
        delete iat_avg;
        delete size_avg;
        for (int i = 0; i < num_states; i++) {
            delete dists_iat[i];
            delete dists_size[i];
        }
        delete dists_size;
        delete dists_iat;
        delete dist_param_size;
        delete dist_param_iat;
    }

    delete rng;
    delete relocate_time_dist;
    delete relocate_state_dist;
}

int JobGenSim::get_curr_index() {
    return 0;
}

JobGenSim *JobGenSim::copy() {
    return new JobGenSim(this);
}
