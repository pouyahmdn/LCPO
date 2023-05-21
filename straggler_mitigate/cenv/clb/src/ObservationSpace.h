//
// Created by Pouya Hamadanian on 5/20/21.
//

#ifndef CLB_OBSERVATIONSPACE_H
#define CLB_OBSERVATIONSPACE_H


class ObservationSpace {
public:
    double *low;
    double *high;
    bool act_in_obs;
    bool load_in_obs;
    bool ext_in_obs;
    bool tw_in_obs;
    bool trace_in_obs;
    unsigned short length;
    int num_servers;

    ObservationSpace(bool act_in_obs, bool load_in_obs, bool ext_in_obs, bool trace_in_obs, bool tw_in_obs,
                     int num_servers);
    ObservationSpace(ObservationSpace* obs_space_base);
    ~ObservationSpace();

    bool contains(const double *observation) const;
};


#endif
