//
// Created by Pouya Hamadanian on 5/19/21.
//

#include <iostream>
#include "LoadBalanceEnv.h"
#include "JobGen.h"
#include "utils.h"
#include <cassert>
#include <limits>
#include "pipes/AgentWindowStatsPipe.h"
#include "pipes/WindowStatsPipe.h"
#include "pipes/FilePipe.h"
#include "pipes/NullPipe.h"
#include "pipes/LoggerSortPipe.h"
#include "pipes/TimeBucketPipe.h"

LoadBalanceEnv::LoadBalanceEnv(unsigned int seed_start, unsigned short num_servers, double time_window,
                               bool load_in_obs, bool ext_in_obs, const std::string &filename_log,
                               bool act_in_obs, bool trace_in_obs, short *timeouts, unsigned short num_timeouts, const double *service_rates,
                               unsigned short len_rates, int max_retries, JobGen *jobGen, bool use_tw,
                               unsigned short unsafety_upper_bound, unsigned short safety_lower_bound, bool skip_log) {
    this->rng = new std::mt19937(seed_start);
    this->num_servers = num_servers;
    this->time_window_ms = time_window;
    this->max_retries = max_retries;
    this -> use_tw = use_tw;
    this -> safety_lower_bound = safety_lower_bound;
    this -> unsafety_upper_bound = unsafety_upper_bound;
    this -> unsafe = false;
    this->wallTime = new WallTime();
    this->timeLine = new TimeLine();
    this->jobGen = jobGen;
    this->server_array = new Server *[num_servers];
    this->server_len = new unsigned short[num_servers]();
    this->average_server_len = new double[num_servers]();
    double rates[num_servers];
    if (len_rates == 1 || len_rates == 2 || len_rates == 2 * this->jobGen->get_num_traces()) {
        if (len_rates == 1)
            this -> avg_rate = service_rates[0];
        else
            this -> avg_rate = jobGen->get_chosen_rate();
        for (int i = 0; i < num_servers; i++)
            rates[i] = this -> avg_rate;
    } else {
        assert (len_rates == num_servers);
        this -> avg_rate = 0;
        for (int i = 0; i < num_servers; i++) {
            rates[i] = service_rates[i];
            this -> avg_rate += service_rates[i];
        }
        this -> avg_rate /= num_servers;
    }
    jobGen->report();
    for (unsigned short i = 0; i < num_servers; i++)
        this->server_array[i] = new Server(i, this->wallTime, this->rng, rates[i]);
    this->last_action = 0;
    this->work_measure_count = new int[MAX_WORK_MEASURE_WINDOWS]();
    this->work_measure_time = new double[MAX_WORK_MEASURE_WINDOWS]();
    this->work_measure_index = 0;
//    this->proc_big_window = new double *[MAX_WINDOWS];
//    this->iat_big_window = new double *[MAX_WINDOWS];
    this->proc_sum_window = new double [MAX_WINDOWS]();
    this->iat_sum_window = new double [MAX_WINDOWS]();
    this->proc_avg_window = new double [MAX_WINDOWS]();
    this->iat_avg_window = new double [MAX_WINDOWS]();
    this->len_big_window = new int[MAX_WINDOWS]();
//    for (unsigned short i = 0; i < MAX_WINDOWS; i++) {
//        this->proc_big_window[i] = new double[MAX_WINDOW_SIZE];
//        this->iat_big_window[i] = new double[MAX_WINDOW_SIZE];
//    }
    this->index_big_window = 0;
    this->size_scale = this->jobGen->get_size_avg() * 10 / this -> avg_rate;
    this->arrival_scale = this->jobGen->get_arrival_avg() * 10;
    this->next_id = 0;
    this->next_job_gen_time = wallTime->curr_time;
    this->next_sim_time_ms = 0;
    this->actionSpace = new ActionSpace(num_timeouts, timeouts);
    this->observationSpace = new ObservationSpace(act_in_obs, load_in_obs, ext_in_obs, trace_in_obs, use_tw, num_servers);

    if (skip_log){
        this->logger = new NullPipe();
    } else {
        auto* agentFile = new FilePipe(filename_log+ "_agent", sizeof(AgentWindowStat));
        auto* allFile = new FilePipe(filename_log+ "_all", sizeof(WindowStat));
        auto* agentStats = new AgentWindowStatsPipe(num_timeouts);
        auto* allStats = new WindowStatsPipe();
        agentStats->appendPipe(agentFile);
        allStats->appendPipe(allFile);
        auto* agentBucket = new TimeBucketPipe(300*1000, true);
        auto* allBucket = new TimeBucketPipe(300*1000, false);
        agentBucket->appendPipe(agentStats);
        allBucket->appendPipe(allStats);
        this->logger = new LoggerSortPipe(20000, true);
        this->logger->appendPipe(agentBucket);
        this->logger->appendPipe(allBucket);
    }
//    this->logger = new Logger(1024 * 1024 * 1024, filename_log, skip_log);
    this->generate_job();
}

LoadBalanceEnv::LoadBalanceEnv(LoadBalanceEnv *base_env) {
    this->rng = new std::mt19937(*(base_env->rng));
    this->num_servers = base_env->num_servers;
    this->time_window_ms = base_env->time_window_ms;
    this->max_retries = base_env->max_retries;
    this -> use_tw = base_env->use_tw;
    this -> safety_lower_bound = base_env->safety_lower_bound;
    this -> unsafety_upper_bound = base_env->unsafety_upper_bound;
    this -> unsafe = base_env->unsafe;
    this->wallTime = new WallTime(base_env->wallTime);
    this->jobGen = base_env->jobGen->copy();

    this->server_len = new unsigned short[num_servers]();
    this->average_server_len = new double[num_servers]();
    for (int i=0; i< num_servers; i++){
        this->server_len[i] = base_env->server_len[i];
        this->average_server_len[i] = base_env->average_server_len[i];
    }
    this->avg_rate = base_env->avg_rate;

    this->server_array = new Server *[num_servers];
    for (unsigned short i = 0; i < num_servers; i++) {
        this->server_array[i] = new Server(base_env->server_array[i], this->rng, this->wallTime);
    }
    this->last_action = base_env->last_action;
    this->index_big_window = base_env->index_big_window;
    this->size_scale = base_env->size_scale;
    this->arrival_scale = base_env->arrival_scale;
    this->next_id = base_env->next_id;
    this->next_job_gen_time = base_env->next_job_gen_time;
    this->next_sim_time_ms = base_env->next_sim_time_ms;
    this->actionSpace = new ActionSpace(base_env->actionSpace);
    this->observationSpace = new ObservationSpace(base_env->observationSpace);
    this->logger = new NullPipe();
//    this->logger = new Logger(1024 * 1024 * 1024, "null", true);
    this->work_measure_index = base_env->work_measure_index;

    this->work_measure_count = new int[MAX_WORK_MEASURE_WINDOWS]();
    this->work_measure_time = new double[MAX_WORK_MEASURE_WINDOWS]();
    for (unsigned short i = 0; i < MAX_WORK_MEASURE_WINDOWS; i++) {
        this->work_measure_count[i] = base_env->work_measure_count[i];
        this->work_measure_time[i] = base_env->work_measure_time[i];
    }
    this->proc_sum_window = new double [MAX_WINDOWS]();
    this->iat_sum_window = new double [MAX_WINDOWS]();
    this->proc_avg_window = new double [MAX_WINDOWS]();
    this->iat_avg_window = new double [MAX_WINDOWS]();
    this->len_big_window = new int[MAX_WINDOWS]();
    for (unsigned short i = 0; i < MAX_WINDOWS; i++) {
        this->proc_sum_window[i] = base_env->proc_sum_window[i];
        this->iat_sum_window[i] = base_env->iat_sum_window[i];
        this->proc_avg_window[i] = base_env->proc_avg_window[i];
        this->iat_avg_window[i] = base_env->iat_avg_window[i];
        this->len_big_window[i] = base_env->len_big_window[i];
    }
    this->timeLine = new TimeLine(base_env->timeLine, this -> num_servers, false);
}

void LoadBalanceEnv::seed(unsigned int seed_start) {
    delete rng;
    rng = new std::mt19937(seed_start);
    for (int i = 0; i < num_servers; i++)
        server_array[i]->set_rng(rng);
}

unsigned short LoadBalanceEnv::best_server(Job *new_job) const {
    unsigned short best, least;
    best = -1;
    least = std::numeric_limits<unsigned short>::max();
    if (new_job->duplicate) {
        for (unsigned short i = 0; i < num_servers; i++)
            if (server_len[i] < least and !new_job->duplicates[i]) {
                least = server_len[i];
                best = i;
            }
    } else {
        for (unsigned short i = 0; i < num_servers; i++)
            if (server_len[i] < least) {
                least = server_len[i];
                best = i;
            }
    }
    return best;
}

void LoadBalanceEnv::set_arrival_scale(double arrival_scale_){
    this -> arrival_scale = arrival_scale_;
}

void LoadBalanceEnv::set_size_scale(double size_scale_){
    this -> size_scale = size_scale_;
}

double LoadBalanceEnv::get_arrival_scale() const{
    return arrival_scale;
}

double LoadBalanceEnv::get_size_scale() const{
    return size_scale;
}

void LoadBalanceEnv::reset_no_obs() {
    for (int i = 0; i < num_servers; i++) {
        server_array[i]->reset();
        server_len[i] = 0;
        average_server_len[i] = 0;
    }
    wallTime->reset();
    timeLine->reset();
    last_action = 0;
    for (unsigned short i = 0; i < MAX_WINDOWS; i++) {
        len_big_window[i] = 0;
        iat_sum_window[i] = 0;
        iat_avg_window[i] = 0;
        proc_sum_window[i] = 0;
        proc_avg_window[i] = 0;
    }
    for (unsigned short i = 0; i < 3; i++) {
        workload_ewma[i] = 0;
        workload_ewma[i+3] = 0;
        workload_ewma_scale[i] = 0;
    }
    index_big_window = 0;
    next_id = 0;
    next_job_gen_time = wallTime->curr_time;
    next_sim_time_ms = 0;
    unsafe = false;
    jobGen->reset();
    generate_job();
}

void LoadBalanceEnv::reset(double *observation) {
    reset_no_obs();
    this->observe(observation);
}

double *LoadBalanceEnv::reset() {
    reset_no_obs();
    return this->observe();
}

void LoadBalanceEnv::generate_job() {
    JobProcessSample sample = jobGen->gen_job();
    Job *new_job = new Job(next_id, sample.size, next_job_gen_time, sample.inter_arrival_time,
                           sample.trace_origin_index);
    auto *jobEvent = new JobEvent{.eventType=EventType(schedule),
                                  .time_key=new_job->arrival_time,
                                  .job=new_job};
    timeLine->push(jobEvent);
    next_id += 1;
    next_job_gen_time += sample.inter_arrival_time;
}

void LoadBalanceEnv::observe(double *observation) {
    unsigned short max_queue = 0;
    for (int i = 0; i < num_servers; i++) {
        observation[i] = server_len[i];
        if (server_len[i] > max_queue)
            max_queue = server_len[i];
        observation[i + num_servers] = average_server_len[i];
    }
    if (use_tw){
        if (unsafe and max_queue <= safety_lower_bound)
            unsafe = false;
        else if (!unsafe and max_queue > unsafety_upper_bound)
            unsafe = true;
    }
    int count = 2 * num_servers;
    if (observationSpace->load_in_obs) {
//        double job_iat_sum = 0;
//        double job_size_sum = 0;
//        int count_jobs = 0;
//        for (int i = 0; i < MAX_WINDOWS; i++) {
//            count_jobs += len_big_window[i];
//            job_iat_sum += sum(iat_big_window[i], len_big_window[i]);
//            job_size_sum += sum(proc_big_window[i], len_big_window[i]);
//        }
        double job_iat_sum = sum(iat_sum_window, MAX_WINDOWS);
        double job_size_sum = sum(proc_sum_window, MAX_WINDOWS);
        int count_jobs = sum(len_big_window, MAX_WINDOWS);
        if (count_jobs == 0) {
            observation[count++] = 0;
            observation[count++] = 0;
            observation[count++] = 0;
            observation[count++] = 0;
            observation[count++] = 0;
        } else {
            observation[count++] = count_jobs / (job_iat_sum + 1e-8) / arrival_scale;
            observation[count++] = job_size_sum / (count_jobs + 1e-8) / size_scale;
//            // This takes the top 50% of inter arrival times and gives us the average. So we only need to deduct this from
//            // the main average to compute the low 50% of inter arrival times.
//            double avg_partition_50_iat = avg_partition_multi_array(iat_big_window, (int *) len_big_window, MAX_WINDOWS,
//                                                                    0.5);
//            observation[count++] =
//                    count_jobs / (2 * job_iat_sum - avg_partition_50_iat * count_jobs + 1e-8) / arrival_scale;
//            double avg_partition_90_size = avg_partition_multi_array(proc_big_window, (int *) len_big_window,
//                                                                     MAX_WINDOWS, 0.1);
//            observation[count++] = avg_partition_90_size / size_scale;
            double job_iat_avg_min = std::numeric_limits<double>::infinity();
            double job_size_avg_max = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < MAX_WINDOWS; ++i) {
                if (job_iat_avg_min > iat_avg_window[i] and len_big_window[i] > 0)
                    job_iat_avg_min = iat_avg_window[i];
                if (job_size_avg_max < proc_avg_window[i] and len_big_window[i] > 0)
                    job_size_avg_max = proc_avg_window[i];
            }
            observation[count++] = 1 / job_iat_avg_min / arrival_scale;
            observation[count++] = job_size_avg_max / size_scale;
            observation[count++] = job_size_sum / (job_iat_sum + 1e-8) / num_servers;
        }
        if (observationSpace->ext_in_obs){
            workload_ewma_scale[0] = 0.9 * workload_ewma_scale[0] + 1;
            workload_ewma_scale[1] = 0.99 * workload_ewma_scale[1] + 1;
            workload_ewma_scale[2] = 0.999 * workload_ewma_scale[2] + 1;
            workload_ewma[0] = 0.9 * workload_ewma[0] + observation[count-5];
            workload_ewma[1] = 0.99 * workload_ewma[1] + observation[count-5];
            workload_ewma[2] = 0.999 * workload_ewma[2] + observation[count-5];
            workload_ewma[3] = 0.9 * workload_ewma[3] + observation[count-4];
            workload_ewma[4] = 0.99 * workload_ewma[4] + observation[count-4];
            workload_ewma[5] = 0.999 * workload_ewma[5] + observation[count-4];
            observation[count++] = workload_ewma[0]/workload_ewma_scale[0];
            observation[count++] = workload_ewma[3]/workload_ewma_scale[0];
            observation[count++] = workload_ewma[1]/workload_ewma_scale[1];
            observation[count++] = workload_ewma[4]/workload_ewma_scale[1];
            observation[count++] = workload_ewma[2]/workload_ewma_scale[2];
            observation[count++] = workload_ewma[5]/workload_ewma_scale[2];
        }
    }


    if (observationSpace->act_in_obs)
        observation[count++] = float(last_action) / float(actionSpace -> n) / 10.0;

    if (observationSpace->trace_in_obs)
        observation[count++] = float(jobGen -> get_curr_trace());

    if (observationSpace->tw_in_obs){
        if (unsafe)
            observation[count++] = 1;
        else
            observation[count++] = 0;
    }

    if (!observationSpace->contains(observation)) {
        for (int i = 0; i < count; i++)
            printf("%.2g\t", observation[i]);
        printf("Observation space violated\n");
        assert(false);
    }
}

double *LoadBalanceEnv::observe() {
    auto observation = new double[observationSpace->length];
    observe(observation);
    return observation;
}

double LoadBalanceEnv::get_avg_rate() const {
    return avg_rate;
}

step_return
LoadBalanceEnv::step(unsigned short action, unsigned short model_index, double *server_time,
                     double *finished_job_duration, double *arrived_job_inter_time, double *arrived_job_proc_time,
                     double* observation) {
    for (int i = 0; i < num_servers; i++){
        server_time[i] = 0;
        average_server_len[i] = 0;
    }
    iat_sum_window[index_big_window] = 0;
    proc_sum_window[index_big_window] = 0;
    unsigned short num_finished_jobs = 0;
    unsigned short num_arrived_jobs = 0;
    double time_start = wallTime->curr_time;
    double time_end = next_sim_time_ms + time_window_ms;
    len_big_window[index_big_window] = 0;
    last_action = action;
    while (timeLine->seek()->time_key <= time_end or num_finished_jobs == 0) {
        JobEvent *jobEvent = timeLine->pop();
        double dt = jobEvent->time_key - wallTime->curr_time;
        for (int i = 0; i < num_servers; i++) {
            if (server_len[i] > 0)
                server_time[i] += dt;
            average_server_len[i] += server_len[i] * dt;
        }
        wallTime->update(jobEvent->time_key);
        if (jobEvent->eventType == finish) {
            server_array[jobEvent->job->assigned_server_id]->len_queue -= 1;
//            jobEvent->job->assigned_server->len_queue -= 1;
            server_len[jobEvent->job->assigned_server_id] -= 1;
//            server_len[jobEvent->job->assigned_server->id] -= 1;
            if (!*(jobEvent->job->completed)) {
                *(jobEvent->job->completed) = true;
                jobEvent->job->first_completed = true;
                assert(jobEvent->job->finish_time == *(jobEvent->job->best_finish_time));
                assert(jobEvent->job->finish_time == wallTime->curr_time);
                finished_job_duration[num_finished_jobs++] = jobEvent->job->get_delay();
            }
            logger->enqueueJob(jobEvent->job);
//            logger->log_job(jobEvent->job);
            jobEvent->job->free();
            delete jobEvent->job;
            delete jobEvent;
        } else if (jobEvent->eventType == schedule) {
            assert(actionSpace->contains(action));
            unsigned short selected_server = best_server(jobEvent->job);
            server_array[selected_server]->schedule(jobEvent->job);
            server_len[selected_server] += 1;
            jobEvent->job->tw_driven = unsafe;
            jobEvent->job->timeout_idx = action;
            jobEvent->job->timeout = actionSpace->timeouts[action];
            jobEvent->job->model_index = model_index;

            jobEvent->eventType = finish;
            jobEvent->time_key = jobEvent->job->finish_time;
            timeLine->push(jobEvent);

            if (!jobEvent->job->duplicate) {
                arrived_job_inter_time[num_arrived_jobs] = jobEvent->job->inter_arrival_time;
                arrived_job_proc_time[num_arrived_jobs++] = jobEvent->job->get_duration();
//                iat_big_window[index_big_window][num_arrived_jobs] = jobEvent->job->inter_arrival_time;
//                proc_big_window[index_big_window][num_arrived_jobs++] = jobEvent->job->get_duration();
                iat_sum_window[index_big_window] += jobEvent->job->inter_arrival_time;
                proc_sum_window[index_big_window] += jobEvent->job->get_duration();
                generate_job();
            }

            if (jobEvent->job->timeout + wallTime->curr_time < *(jobEvent->job->best_finish_time) and
                *(jobEvent->job->num_instances) <= max_retries and jobEvent->job->timeout > 0 and
                *(jobEvent->job->num_instances) < num_servers) {
                Job *dup_job = new Job(jobEvent->job, num_servers);
                auto *dup_jobEvent = new JobEvent{.eventType=schedule,
                        .time_key=jobEvent->job->timeout + wallTime->curr_time,
                        .job=dup_job};
                timeLine->push(dup_jobEvent);
            }
        }
    }
    double time_elapsed = wallTime->curr_time - time_start;

    if (time_elapsed > 0) {
        for (int i = 0; i < num_servers; i++)
            average_server_len[i] /= time_elapsed;
    }
    else {
        for (int i = 0; i < num_servers; i++)
            average_server_len[i] = server_len[i];
    }

    work_measure_time[work_measure_index] = iat_sum_window[index_big_window];
    work_measure_count[work_measure_index] = num_arrived_jobs;
    work_measure_index = (work_measure_index + 1) % MAX_WORK_MEASURE_WINDOWS;

    len_big_window[index_big_window] = num_arrived_jobs;
    iat_avg_window[index_big_window] = iat_sum_window[index_big_window] / (num_arrived_jobs + 1e-8);
    proc_avg_window[index_big_window] = proc_sum_window[index_big_window] / (num_arrived_jobs + 1e-8);
    double step_proc_avg = proc_avg_window[index_big_window];
    double step_iat_avg = iat_avg_window[index_big_window];
    index_big_window = (index_big_window + 1) % MAX_WINDOWS;

    if (wallTime->curr_time <= time_end)
        next_sim_time_ms += time_window_ms;
    else
        next_sim_time_ms += time_elapsed;

    bool done = true;
    for (int i = 0; i < num_servers; i++)
        if (server_len[i] > 0)
            done = false;

    double rew_p = percentile(finished_job_duration, num_finished_jobs, 0.95);

    observe(observation);

    return step_return{
            .num_finished_jobs=num_finished_jobs,
            .done=done,
            .next_obs=observation,
            .server_time=server_time,
            .time_elapsed=time_elapsed,
            .finished_job_completion_times=finished_job_duration,
            .reward=rew_p,
            .arrived_job_inter_time=arrived_job_inter_time,
            .arrived_job_proc_time=arrived_job_proc_time,
            .avg_arrived_job_inter_time=step_iat_avg,
            .avg_arrived_job_proc_time=step_proc_avg,
            .num_arrived_jobs=num_arrived_jobs,
            .curr_time=wallTime->curr_time,
            .unsafe=unsafe
    };
}

step_return LoadBalanceEnv::step(unsigned short action, unsigned short model_index) {
    auto *server_time = new double[num_servers]();
    auto *finished_job_duration = new double[MAX_WINDOW_SIZE];
    auto *arrived_job_inter_time = new double[MAX_WINDOW_SIZE];
    auto *arrived_job_proc_time = new double[MAX_WINDOW_SIZE];
    auto observation = new double[observationSpace->length];
    return step(action, model_index, server_time, finished_job_duration, arrived_job_inter_time,
                arrived_job_proc_time, observation);
}

void LoadBalanceEnv::close() const {
    logger->flush();
}

LoadBalanceEnv::~LoadBalanceEnv() {
    delete rng;
    delete wallTime;
    delete timeLine;
    for (int i = 0; i < num_servers; i++)
        delete server_array[i];
    delete server_array;
    delete observationSpace;
    delete actionSpace;
    delete logger;
    delete server_len;
    delete average_server_len;
    delete work_measure_count;
    delete work_measure_time;
    delete len_big_window;
//    for (int i = 0; i < MAX_WINDOWS; i++) {
//        delete proc_big_window[i];
//        delete iat_big_window[i];
//    }
//    delete proc_big_window;
//    delete iat_big_window;
    delete proc_avg_window;
    delete iat_avg_window;
    delete proc_sum_window;
    delete iat_sum_window;
}

double LoadBalanceEnv::get_work_measure() const {
    return sum(work_measure_count, MAX_WORK_MEASURE_WINDOWS) / sum(work_measure_time, MAX_WORK_MEASURE_WINDOWS) * 1000;
}

unsigned int LoadBalanceEnv::observation_len() const {
    return observationSpace->length;
}

unsigned int LoadBalanceEnv::timeline_len() const {
    return timeLine->len();
}

void LoadBalanceEnv::queue_sizes(unsigned short *queues) const {
    for (int i = 0; i < num_servers; i++)
        queues[i] = server_len[i];
}

unsigned short *LoadBalanceEnv::queue_sizes() const {
    auto* queues = new unsigned short[num_servers];
    queue_sizes(queues);
    return queues;
}
