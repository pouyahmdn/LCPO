//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_NULLPIPE_H
#define CLB_NULLPIPE_H

#include "pipe.h"

class NullPipe: public Pipe {
public:
    NullPipe() = default;
    ~NullPipe() override = default;

    void enqueueJob(Job* job) override {};
    void enqueue(void* entry) override {};
    void flush() override {};
};

#endif