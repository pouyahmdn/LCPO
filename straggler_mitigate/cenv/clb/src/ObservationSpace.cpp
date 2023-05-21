//
// Created by Pouya Hamadanian on 5/20/21.
//

#include "ObservationSpace.h"
#include "iostream"

ObservationSpace::ObservationSpace(bool act_in_obs, bool load_in_obs, bool ext_in_obs, bool trace_in_obs,
                                   bool tw_in_obs, int num_servers) : act_in_obs(act_in_obs), load_in_obs(load_in_obs),
                                   tw_in_obs(tw_in_obs), trace_in_obs(trace_in_obs), ext_in_obs(ext_in_obs) {
    this->length = 2 * num_servers;
    if (load_in_obs){
        this->length += 5;
        if (ext_in_obs)
            this->length += 6;
    }
    if (act_in_obs)
        this->length += 1;
    if (tw_in_obs)
        this->length += 1;
    if (trace_in_obs)
        this->length += 1;
    this->low = new double[this->length];
    this->high = new double[this->length];
    this -> num_servers = num_servers;

    for (int i = 0; i < this->length; i++)
        this->low[i] = 0;
    int count;
    for (count = 0; count < 2 * num_servers; count++)
        this->high[count] = 10000;
    if (load_in_obs) {
        this->high[count++] = 10;
        this->high[count++] = 10;
//        high[count++] = 1000;
//        high[count++] = 5000;
        this->high[count++] = 40;
        this->high[count++] = 40;
        this->high[count++] = 10;
        if (ext_in_obs) {
            this->high[count++] = 10;
            this->high[count++] = 10;
            this->high[count++] = 10;
            this->high[count++] = 10;
            this->high[count++] = 10;
            this->high[count++] = 10;
        }
    }
    if (act_in_obs)
        this->high[count++] = 0.1;
    if (trace_in_obs)
        this->high[count++] = 10;
    if (tw_in_obs)
        this->high[count++] = 1;
}

ObservationSpace::ObservationSpace(ObservationSpace *obs_space_base) {
    this -> load_in_obs = obs_space_base->load_in_obs;
    this -> ext_in_obs = obs_space_base->ext_in_obs;
    this -> act_in_obs = obs_space_base->act_in_obs;
    this -> tw_in_obs = obs_space_base->tw_in_obs;
    this -> trace_in_obs = obs_space_base->trace_in_obs;
    this->length = obs_space_base->length;
    this->low = new double[this->length];
    this->high = new double[this->length];
    this -> num_servers = obs_space_base->num_servers;

    for (int i = 0; i < this->length; i++){
        this->low[i] = obs_space_base->low[i];
        this->high[i] = obs_space_base->high[i];
    }
}

bool ObservationSpace::contains(const double *observation) const {
    bool contained = true;
    for (int i = 0; i < length; i++)
        contained = contained and (observation[i] <= high[i]) and (observation[i] >= low[i]);
    return contained;
}

ObservationSpace::~ObservationSpace() {
    delete high;
    delete low;
}
