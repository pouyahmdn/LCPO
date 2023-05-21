//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_TIMELINE_H
#define CLB_TIMELINE_H

#include "Job.h"
#include "queue"

enum EventType{
    schedule, finish
};

struct JobEvent{
    EventType eventType;
    double time_key;
    Job* job;
};

struct CompareJobEvent {
    bool operator()(const JobEvent* lhs, const JobEvent* rhs) const{
        return lhs->time_key > rhs -> time_key or (lhs->time_key == rhs -> time_key and lhs -> eventType == schedule and rhs -> eventType == finish);
    }
};

class TimeLine {
private:
    std::priority_queue<JobEvent*, std::vector<JobEvent*>, CompareJobEvent>* pq;

public:

    TimeLine();
    TimeLine(TimeLine* line_base, unsigned short num_servers, bool filter_first_schedule);
    ~TimeLine();
    void push(JobEvent* jobEvent);
    JobEvent* pop();
    JobEvent* seek();
    unsigned int len();
    void reset();
    double get_schedule_inter_time();
};


#endif
