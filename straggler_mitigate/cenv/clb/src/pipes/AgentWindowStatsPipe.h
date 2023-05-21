//
// Created by Pouya Hamadanian on 8/10/22.
//

#ifndef CLB_AGENTWINDOWSTATSPIPE_H
#define CLB_AGENTWINDOWSTATSPIPE_H

#include "pipe.h"
#include <vector>
#include "WindowStatsPipe.h"

struct AgentWindowStat{
    SingleStat<float> arrival;
    SingleStat<float> action;
    unsigned int histogram[10];
    float entropy;
    float interval_duration;
    size_t len;
    unsigned char trace;
};

class AgentWindowStatsPipe: public Pipe {
public:
    explicit AgentWindowStatsPipe(int action_count);
    ~AgentWindowStatsPipe() override = default;

    void flush() override;
    void enqueue(void* entry) override;
    void extend(std::vector<void*>& arr_entry) override;
};

#endif //CLB_AGENTWINDOWSTATSPIPE_H
