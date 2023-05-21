//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_TIMEBUCKETPIPE_H
#define CLB_TIMEBUCKETPIPE_H

#include "pipe.h"
#include <vector>

class TimeBucketPipe: public Pipe {
private:
    float bucketInterval;
    float lastStart;
    bool twFilter;
    unsigned char lastTrace;
    std::vector<void*> bucket;

    void popAndForward();

public:
    TimeBucketPipe(float bucketInterval, bool twFilter);
    ~TimeBucketPipe() override;

    void flush() override;
    void enqueue(void* entry) override;
};

#endif