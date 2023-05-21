//
// Created by Pouya Hamadanian on 5/24/21.
//

#ifndef CLB_STATIC_DIST_H
#define CLB_STATIC_DIST_H

#include "distribution.h"

class static_dist : public distribution
{
private:
    const double _ret_val_;

public:
    // constructors and reset functions
    explicit static_dist(const double ret_val) : _ret_val_(ret_val) {}

    // generating functions
    double generate(std::mt19937* _g) override;

    // average functions
    double average() override;
    double average(double bound_low, double bound_high) override;
    static_dist* copy() override;
};

#endif //CLB_STATIC_DIST_H
