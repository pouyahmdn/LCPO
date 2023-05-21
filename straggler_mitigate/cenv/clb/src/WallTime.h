//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_WALLTIME_H
#define CLB_WALLTIME_H


class WallTime {
public:
    double curr_time;

    WallTime();
    explicit WallTime(WallTime* time_base);
    ~WallTime();

    void update(double new_time);
    void reset();
};


#endif
