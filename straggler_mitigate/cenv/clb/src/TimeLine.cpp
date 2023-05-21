//
// Created by Pouya Hamadanian on 5/19/21.
//

#include <queue>
#include "TimeLine.h"
#include "unordered_map"
#include "iostream"
#include <cassert>

TimeLine::TimeLine() {
    this -> pq = new std::priority_queue<JobEvent*, std::vector<JobEvent*>, CompareJobEvent>;
}

void TimeLine::push(JobEvent* jobEvent) {
    pq -> push(jobEvent);
}

JobEvent* TimeLine::pop() {
    JobEvent* ret_pop = pq -> top();
    pq -> pop();
    return ret_pop;
}

JobEvent* TimeLine::seek() {
    return pq -> top();
}

void TimeLine::reset() {
    while(!pq -> empty()){
        JobEvent* ret_pop = pq -> top();
        pq -> pop();
        ret_pop->job->free();
        delete ret_pop -> job;
        delete ret_pop;
    }
}

unsigned int TimeLine::len() {
    return pq->size();
}

TimeLine::~TimeLine() {
    reset();
    delete pq;
}

std::vector<JobEvent*>& Container(std::priority_queue<JobEvent*, std::vector<JobEvent*>, CompareJobEvent>* q) {
    struct HackedQueue : private std::priority_queue<JobEvent*, std::vector<JobEvent*>, CompareJobEvent> {
        static std::vector<JobEvent*>& Container(std::priority_queue<JobEvent*, std::vector<JobEvent*>, CompareJobEvent>* q) {
            return q->*&HackedQueue::c;
        }
    };
    return HackedQueue::Container(q);
}

struct JobRep {
    int reps_left;
    Job* prev_inst;
};

TimeLine::TimeLine(TimeLine *line_base, unsigned short num_servers, bool filter_first_schedule) {
    this -> pq = new std::priority_queue<JobEvent*, std::vector<JobEvent*>, CompareJobEvent>;
    std::vector<JobEvent*> &jobs = Container(line_base->pq);
    std::unordered_map<unsigned int, JobRep*> instance_rep_map;
    bool saw_first_schedule = false;
    for(auto job_iter: jobs){
        JobEvent* new_job_event;
        if (filter_first_schedule and job_iter->eventType == schedule and !job_iter->job->duplicate){
            assert(!saw_first_schedule);
            saw_first_schedule = true;
            continue;
        }
        if (instance_rep_map.find(job_iter->job->id) == instance_rep_map.end()){    // Job was not previously added to timeline
            new_job_event = new JobEvent{
                .eventType=job_iter->eventType,
                .time_key=job_iter->time_key,
                .job=new Job(job_iter->job, nullptr, num_servers)};
            if (*(job_iter->job->active_instances) > 1){
                instance_rep_map[job_iter->job->id] = new JobRep{
                    .reps_left=*(job_iter->job->active_instances)-1,
                    .prev_inst=new_job_event->job
                };
            }
        } else {                                                                    // Job was previously added to timeline and is available at the hash map
            new_job_event = new JobEvent{
                    .eventType=job_iter->eventType,
                    .time_key=job_iter->time_key,
                    .job=new Job(job_iter->job, instance_rep_map[job_iter->job->id]->prev_inst, num_servers)};
            instance_rep_map[job_iter->job->id]->reps_left -= 1;
            if (instance_rep_map[job_iter->job->id]->reps_left == 0) {
                delete instance_rep_map[job_iter->job->id];
                instance_rep_map.erase(job_iter->job->id);
            }
        }
        this-> pq-> push(new_job_event);
    }
    if (!instance_rep_map.empty()){
        std::cout << "Fatal flaw while copying timeline: Repetition Hash Map did not empty" << std::endl;
        assert(false);
    }
}

double TimeLine::get_schedule_inter_time() {
    std::vector<JobEvent*> &jobs = Container(pq);
    bool saw_first_schedule = false;
    double inter_time_schedule = -1;
    for(auto job_iter: jobs){
        if (job_iter->eventType == schedule and !job_iter->job->duplicate){
            assert(!saw_first_schedule);
            saw_first_schedule = true;
            inter_time_schedule = job_iter->job->inter_arrival_time;
        }
    }
    assert(saw_first_schedule);
    return inter_time_schedule;
}
