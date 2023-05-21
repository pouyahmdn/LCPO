//
// Created by Pouya Hamadanian on 8/9/22.
//
#include "TimeBucketPipe.h"

TimeBucketPipe::TimeBucketPipe(float bucketInterval, bool twFilter){
    this->bucketInterval=bucketInterval;
    this->lastStart = 0;
    this->lastTrace = 0;
    this->twFilter = twFilter;
}

void TimeBucketPipe::popAndForward() {
    for (Pipe* pipe: next_pipes)
        pipe->extend(bucket);
    for (auto& ret_pop: bucket) {
        auto* logEntry = (LogEntry*) ret_pop;
        if (logEntry->taps == 1)
            delete logEntry;
        else
            logEntry->taps -= 1;
    }
    bucket.clear();
}

void TimeBucketPipe::enqueue(void* entry){
    auto* logEntry = (LogEntry*) entry;
    if (!twFilter || !logEntry->tw){
        logEntry->taps += 1;
        if (!bucket.empty() && (logEntry->arrival - lastStart > bucketInterval || logEntry->trace_index != lastTrace))
            popAndForward();
        if (bucket.empty()){
            lastStart = logEntry->arrival;
            lastTrace = logEntry->trace_index;
        }
        bucket.push_back(entry);
    }
}

TimeBucketPipe::~TimeBucketPipe(){
    if (!bucket.empty())
        popAndForward();
    for (Pipe *pipe: next_pipes)
        pipe->flush();
}

void TimeBucketPipe::flush(){
    if (!bucket.empty())
        popAndForward();
    for (Pipe *pipe: next_pipes)
        pipe->flush();
}