
#include "pipe.h"

LogEntry* Pipe::translateEntry(Job* job){
    return new LogEntry{
            .arrival=(float) job->arrival_time,
            .delay=(float) job->get_delay(),
            .size=(float) job->original_duration,
            .duration= (float) job->get_duration(),
            .first_duration=(float) job->get_first_duration(),
            .id=job->id,
            .timeout_idx=job->timeout_idx,
            .first=job->first_completed,
            .tw=job->tw_driven,
            .num_instances=*(job->num_instances),
            .instance_index=job->instance_index,
            .queue_obs=job->queue_observed,
            .queue_obs_first=job->get_first_queue_obs(),
            .model_index=job->model_index,
            .trace_index=job->trace_origin_index,
            .taps=0,
    };
}

void Pipe::enqueueJob(Job* job){
    enqueue((void*)translateEntry(job));
}

void Pipe::extend(std::vector<void*>& arr_entry){
    for (auto& entry: arr_entry)
        enqueue(entry);
}

void Pipe::appendPipe(Pipe* forward_pipe){
    next_pipes.push_back(forward_pipe);
}