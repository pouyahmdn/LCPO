//
// Created by Pouya Hamadanian on 8/10/22.
//

#include "iostream"
#include "AgentWindowStatsPipe.h"
#include <numeric>
#include <cmath>

float entropy(std::vector<unsigned int> hist){
    float sum=(float)std::accumulate(hist.begin(), hist.end(), 0.0), ent=0;
    for(unsigned int h : hist){
        if (h == 0)
            continue;
        float prob = (float)h / sum;
        ent -= prob * log(prob);
    }
    return ent;
}

AgentWindowStatsPipe::AgentWindowStatsPipe(int action_count){
    if (action_count > 10){
        std::cout << "Action count is more than 10, exiting..." << std::endl;
        exit(1);
    }
}

void AgentWindowStatsPipe::flush(){
    for (auto& pipe: next_pipes)
        pipe->flush();
}

void AgentWindowStatsPipe::enqueue(void* entry){
    std::cout << "Called enqueue on stat calculating pipe, exiting..." << std::endl;
    exit(1);
}

void AgentWindowStatsPipe::extend(std::vector<void*>& arr_entry){
    // Length
    auto* agentWindowStat = new AgentWindowStat{
            .histogram={0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            .len=arr_entry.size(),
    };

    if (agentWindowStat->len > 0){
        // Trace index
        agentWindowStat->trace = ((LogEntry*)arr_entry.front())->trace_index;
        // Bucket duration
        agentWindowStat->interval_duration = ((LogEntry*)arr_entry.back())->arrival - ((LogEntry*)arr_entry.front())->arrival;

        float data[2][agentWindowStat->len];

        for (std::vector<void*>::size_type i = 0; i != arr_entry.size(); i++){
            auto* logEntry = (LogEntry*) arr_entry[i];

            data[0][i] = logEntry->arrival;
            data[1][i] = logEntry->timeout_idx;

            // Histogram
            agentWindowStat->histogram[logEntry->timeout_idx] += 1;
        }

        // 2 Single stats
        agentWindowStat->arrival = get_single_stat(std::vector<float>(data[0], data[0]+agentWindowStat->len), false);
        agentWindowStat->action = get_single_stat(std::vector<float>(data[1], data[1]+agentWindowStat->len), true);

        agentWindowStat->entropy = entropy(std::vector<unsigned int>(agentWindowStat->histogram, agentWindowStat->histogram+10));
    }

    for (Pipe* pipe: next_pipes)
        pipe->enqueue((void*)agentWindowStat);
    delete agentWindowStat;
}