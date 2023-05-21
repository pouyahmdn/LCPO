//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_LOGGERSORTPIPE_H
#define CLB_LOGGERSORTPIPE_H

#include "queue"
#include "pipe.h"
#include <vector>

struct CompareLogEntry {
    bool operator()(const LogEntry* lhs, const LogEntry* rhs) const{
        return lhs -> id > rhs -> id;
    }
};

class LoggerSortPipe: public Pipe {
private:
    unsigned int size_heap;
    unsigned int id;
    bool valid;
    std::priority_queue<LogEntry*, std::vector<LogEntry*>, CompareLogEntry>* pq;
    void popAndForward();

public:
    LoggerSortPipe(unsigned int size_heap, bool valid);
    ~LoggerSortPipe() override;

    void flush() override;
    void enqueue(void* entry) override;
};

#endif