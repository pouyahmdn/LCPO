//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_JOB_H
#define CLB_JOB_H

class Server;

class Job {

public:
    double inter_arrival_time;
    double arrival_time;
    unsigned int id;
    double size;
    double enqueue_time{};
    double start_time{};
    double finish_time{};
    double duration_first;
    double original_duration{};
    unsigned char queue_observed{};
    unsigned char queue_observed_first{};

    unsigned char timeout_idx{};
    short timeout{};
    unsigned char instance_index;
    unsigned char model_index{};
    unsigned char trace_origin_index;
    bool tw_driven{};

    bool *completed;
    unsigned char *num_instances;
    unsigned char *active_instances;
    double *best_finish_time;
    bool first_completed;

    bool *duplicates;
    bool duplicate;

    unsigned short assigned_server_id{};

    Job(unsigned int id, double size, double arrival_time, double inter_arrival_time,
        unsigned char trace_origin_index);
    Job(Job *base, Job *prev_instance, unsigned short num_servers);
    Job(Job *base, unsigned short num_servers);
    ~Job();
    double get_duration() const;
    double get_first_duration() const;
    unsigned char get_first_queue_obs() const;
    double get_delay() const;
    void free();
};


#endif
