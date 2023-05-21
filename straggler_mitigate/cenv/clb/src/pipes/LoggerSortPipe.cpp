//
// Created by Pouya Hamadanian on 8/9/22.
//

#include "LoggerSortPipe.h"
#include "iostream"

LoggerSortPipe::LoggerSortPipe(unsigned int size_heap, bool valid) {
    this->pq = new std::priority_queue<LogEntry *, std::vector<LogEntry *>, CompareLogEntry>;
    this->size_heap = size_heap;
    this->id = 0;
    this->valid = valid;
}

void LoggerSortPipe::popAndForward() {
    LogEntry* ret_pop = pq -> top();
    pq -> pop();
    if (ret_pop->id < id){
        std::cout << "Log Sorter failed to sort, last ID was " << id << ", new ID was " << ret_pop->id << std::endl;
        exit(1);
    } else {
        id = ret_pop->id;
    }
    for (Pipe* pipe: next_pipes)
        pipe->enqueue((void*)ret_pop);
    if (ret_pop->taps == 1)
        delete ret_pop;
    else
        ret_pop->taps -= 1;
}

void LoggerSortPipe::enqueue(void *entry) {
    auto* logEntry = (LogEntry*) entry;
    if (!valid || logEntry->first) {
        logEntry->taps += 1;
        pq -> push(logEntry);
        while (pq -> size() >= size_heap)
            popAndForward();
    }
}

void LoggerSortPipe::flush(){
    while(!pq -> empty())
        popAndForward();
    for (Pipe *pipe: next_pipes)
        pipe->flush();
}

LoggerSortPipe::~LoggerSortPipe(){
    while(!pq -> empty())
        popAndForward();
    for (Pipe *pipe: next_pipes)
        pipe->flush();
    delete pq;
}