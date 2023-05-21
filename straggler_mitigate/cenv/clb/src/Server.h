//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_SERVER_H
#define CLB_SERVER_H

#include "WallTime.h"
#include <random>

class Job;

class Server {
private:
    double PROB_INFLATION = 0.1;
    double service_rate;
    WallTime* wallTime;
    double finish_time;
    std::mt19937* rng;
    std::bernoulli_distribution* distribution;

public:
    unsigned short id;
    unsigned short len_queue;

    Server(unsigned short id, WallTime* wallTime, std::mt19937* rng, double service_rate);
    explicit Server(Server* base_server, std::mt19937* rng, WallTime* wallTime);
    ~Server();
    void schedule(Job* job);
    void reset();
    void set_rng(std::mt19937* rng_);
};


#endif //CLB_SERVER_H
