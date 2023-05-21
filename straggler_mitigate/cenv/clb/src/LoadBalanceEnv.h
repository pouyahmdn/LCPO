//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_LOADBALANCEENV_H
#define CLB_LOADBALANCEENV_H

#include<random>
#include "WallTime.h"
#include "TimeLine.h"
#include "Server.h"
#include "ObservationSpace.h"
#include "ActionSpace.h"
//#include "Logger.h"
#include "pipes/pipe.h"
#include "JobGen.h"

struct step_return {
    unsigned short num_finished_jobs;
    bool done;
    double *next_obs;
    double *server_time;
    double time_elapsed;
    double *finished_job_completion_times;
    double reward;
    double *arrived_job_inter_time;
    double *arrived_job_proc_time;
    double avg_arrived_job_inter_time;
    double avg_arrived_job_proc_time;
    unsigned short num_arrived_jobs;
    double curr_time;
    bool unsafe;
};

class LoadBalanceEnv {
public:
    const int MAX_WINDOW_SIZE = 10000;
    const int MAX_WINDOWS = 4;
    const int MAX_WORK_MEASURE_WINDOWS = 20;
    std::mt19937 *rng;
    WallTime *wallTime;
    TimeLine *timeLine;
    Server **server_array;
    ObservationSpace *observationSpace;
    ActionSpace *actionSpace;
    Pipe* logger;
    JobGen* jobGen;
    unsigned short *server_len;
    double *average_server_len;
    unsigned short num_servers;
    unsigned short last_action;

//    double **proc_big_window;
//    double **iat_big_window;
    double *proc_sum_window;
    double *iat_sum_window;
    double *proc_avg_window;
    double *iat_avg_window;
    int *len_big_window;
    int index_big_window;
    int *work_measure_count;
    double *work_measure_time;
    int work_measure_index;
    double time_window_ms;
    int max_retries;
    double size_scale;
    double arrival_scale;
    unsigned int next_id;
    double next_job_gen_time;
    double next_sim_time_ms;

    double workload_ewma[6] = {0, 0, 0, 0, 0, 0};
    double workload_ewma_scale[3] = {0, 0, 0};

    bool use_tw;
    unsigned short unsafety_upper_bound;
    unsigned short safety_lower_bound;
    bool unsafe;
    double avg_rate;

public:
    LoadBalanceEnv(unsigned int seed_start, unsigned short num_servers, double time_window, bool load_in_obs,
                   bool ext_in_obs, const std::string& filename_log,  bool act_in_obs, bool trace_in_obs,
                   short *timeouts, unsigned short num_timeouts, const double *service_rates,
                   unsigned short len_rates, int max_retries, JobGen* jobGen, bool use_tw,
                   unsigned short unsafety_upper_bound, unsigned short safety_lower_bound, bool skip_log);

    explicit LoadBalanceEnv(LoadBalanceEnv* base_env);

    ~LoadBalanceEnv();

    void seed(unsigned int seed_start);

    void set_arrival_scale(double arrival_scale);

    void set_size_scale(double size_scale);

    double get_arrival_scale() const;

    double get_size_scale() const;

    double *reset();

    void reset(double* observation);

    void generate_job();

    void queue_sizes(unsigned short* queues) const;

    double get_avg_rate() const;

    unsigned short* queue_sizes() const;

    unsigned int observation_len() const;

    unsigned int timeline_len() const;

    double* observe();

    void observe(double* observation) ;

    void close() const;

    unsigned short best_server(Job *new_job) const;

    double get_work_measure() const;

    step_return step(unsigned short action, unsigned short model_index);

    step_return step(unsigned short action, unsigned short model_index, double* server_time,
                     double* finished_job_duration, double* arrived_job_inter_time, double* arrived_job_proc_time,
                     double* observation);

    void reset_no_obs();
};

#endif
