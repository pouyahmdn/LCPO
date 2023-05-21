//
// Created by Pouya Hamadanian on 5/19/21.
//

#include "Job.h"
#include "Server.h"
#include <limits>

Job::Job(unsigned int id, double size, double arrival_time, double inter_arrival_time,
         unsigned char trace_origin_index) {
    this -> id = id;
    this -> size = size;
    this -> arrival_time = arrival_time;
    this -> inter_arrival_time = inter_arrival_time;
    this -> trace_origin_index = trace_origin_index;
    this -> instance_index = 0;
    this -> duration_first = 0;

    this -> completed = new bool(false);
    this -> first_completed = false;
    this -> num_instances = new unsigned char (1);
    this -> active_instances = new unsigned char (1);
    this -> best_finish_time = new double (std::numeric_limits<double>::infinity());
    this -> duplicate = false;
    this -> duplicates = nullptr;
}

Job::Job(Job* base, Job *prev_instance, unsigned short num_servers) {
    this -> id = base->id;
    this -> size = base->size;
    this -> enqueue_time = base->enqueue_time;
    this -> start_time = base->start_time;
    this -> finish_time = base->finish_time;
    this -> duration_first = base->duration_first;
    this -> original_duration = base->original_duration;
    this -> queue_observed = base->queue_observed;
    this -> queue_observed_first = base->queue_observed_first;
    this -> timeout_idx = base->timeout_idx;
    this -> timeout = base->timeout;
    this -> instance_index = base->instance_index;
    this -> model_index = base->model_index;
    this -> trace_origin_index = base->trace_origin_index;
    this -> tw_driven = base->tw_driven;
    this -> duplicate = base->duplicate;
    this -> assigned_server_id = base->assigned_server_id;
    this -> arrival_time = base->arrival_time;
    this -> inter_arrival_time = base->inter_arrival_time;
    this -> first_completed = base->first_completed;

    if (prev_instance == nullptr) {
        this->completed = new bool(*(base->completed));
        this->num_instances = new unsigned char(*(base->num_instances));
        this->active_instances = new unsigned char(*(base->active_instances));
        this->best_finish_time = new double(*(base->best_finish_time));

        if (base->duplicates == nullptr)
            this->duplicates = nullptr;
        else {
            this->duplicates = new bool[num_servers];
            for (int i=0; i< num_servers; i++)
                this -> duplicates[i] = base->duplicates[i];
        }
    } else {
        this->completed = prev_instance->completed;
        this->num_instances = prev_instance->num_instances;
        this->active_instances = prev_instance->active_instances;
        this->best_finish_time = prev_instance->best_finish_time;
        this->duplicates = prev_instance->duplicates;
    }
}

Job::Job(Job* base, unsigned short num_servers) {
    this -> id = base->id;
    this -> size = base->size;
    this -> trace_origin_index = base->trace_origin_index;
    this -> duration_first = base->get_first_duration();
    this -> queue_observed_first = base->queue_observed;
    this -> arrival_time = base->arrival_time;
    this -> inter_arrival_time = base->inter_arrival_time;
    this -> instance_index = base->instance_index + 1;
    this -> completed = base->completed;
    this -> num_instances = base->num_instances;
    this -> active_instances = base->active_instances;
    this -> best_finish_time = base->best_finish_time;
    *(this->num_instances) += 1;
    *(this->active_instances) += 1;
    this -> first_completed = false;
    if (base ->duplicate)
        this -> duplicates = base->duplicates;
    else{
        this -> duplicates = new bool[num_servers]{false};
        base -> duplicates = this -> duplicates;
    }
    this -> duplicates[base->assigned_server_id] = true;
    this -> duplicate = true;
}

Job::~Job() = default;

double Job::get_duration() const {
    return finish_time-start_time;
}

double Job::get_first_duration() const {
    if (this -> duplicate)
        return this -> duration_first;
    else
        return finish_time-start_time;
}

unsigned char Job::get_first_queue_obs() const {
    if (this -> duplicate)
        return this -> queue_observed_first;
    else
        return this -> queue_observed;
}

double Job::get_delay() const {
    return finish_time-arrival_time;
}

void Job::free() {
    *(active_instances) -= 1;
    if (*active_instances == 0){
        delete completed;
        delete num_instances;
        delete active_instances;
        delete best_finish_time;
        delete duplicates;
    }
}
