//
// Created by Pouya Hamadanian on 5/19/21.
//

#include "Server.h"
#include "Job.h"
#include <random>

Server::Server(unsigned short id, WallTime* wallTime, std::mt19937* rng, double service_rate) {
    this -> id = id;
    this -> wallTime = wallTime;
    this -> len_queue = 0;
    this -> finish_time = wallTime->curr_time;
    this -> service_rate = service_rate;
    this -> rng = rng;
    this -> distribution = new std::bernoulli_distribution(PROB_INFLATION);
}

Server::Server(Server *base_server, std::mt19937* rng, WallTime* wallTime) {
    this -> id = base_server->id;
    this -> wallTime = wallTime;
    this -> len_queue = base_server->len_queue;
    this -> finish_time = base_server->finish_time;
    this -> service_rate = base_server->service_rate;
    this -> distribution = new std::bernoulli_distribution(PROB_INFLATION);
    this -> rng = rng;
}

Server::~Server(){
    delete distribution;
}

void Server::schedule(Job* job) {
    double duration = job->size / service_rate;
    job->original_duration = duration;
    job->queue_observed = len_queue;
    if ((*distribution)(*(rng)))
        duration *= 10;
    if (len_queue > 0){
        job->finish_time = finish_time + duration;
        job->start_time = finish_time;
    }
    else{
        job->finish_time = wallTime -> curr_time + duration;
        job->start_time = wallTime -> curr_time;
    }
    finish_time = job->finish_time;
    if (*(job->best_finish_time) > job->finish_time){
        *(job->best_finish_time) = job->finish_time;
    }
    job->enqueue_time = wallTime -> curr_time;
    len_queue += 1;
    job->assigned_server_id = id;
}

void Server::reset() {
    len_queue = 0;
}

void Server::set_rng(std::mt19937 *rng_) {
    this -> rng = rng_;
}
