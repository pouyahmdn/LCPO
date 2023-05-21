//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_PIPE_H
#define CLB_PIPE_H

#include "../Job.h"
#include <vector>

struct LogEntry{
    float arrival;
    float delay;
    float size;
    float duration;
    float first_duration;
    unsigned int id;
    unsigned char timeout_idx;
    bool first;
    bool tw;
    unsigned char num_instances;
    unsigned char instance_index;
    unsigned char queue_obs;
    unsigned char queue_obs_first;
    unsigned char model_index;
    unsigned char trace_index;

    unsigned char taps;
};

class Pipe {
protected:
    std::vector<Pipe*> next_pipes;

public:
    virtual ~Pipe() = default;
    virtual void enqueue(void* entry) = 0;
    virtual void flush() = 0;

    static LogEntry* translateEntry(Job* job);
    virtual void enqueueJob(Job* job);
    virtual void extend(std::vector<void*>& arr_entry);
    void appendPipe(Pipe* forward_pipe);
};


#endif