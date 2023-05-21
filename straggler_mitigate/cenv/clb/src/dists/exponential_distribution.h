//
// Created by Pouya Hamadanian on 5/24/21.
//

#ifndef CLB_PARETO_DISTRIBUTION_H
#define CLB_PARETO_DISTRIBUTION_H

#include "distribution.h"

class exponential_dist : public distribution
{
private:
    const double _rev_lambda_;

public:
    // constructors and reset functions
    explicit exponential_dist(double lambda = 1): _rev_lambda_(lambda) {}

    // generating functions
    double generate(std::mt19937* _g) override;

    // average functions
    double average() override;
    double average(double bound_low, double bound_high) override;
    exponential_dist* copy() override;
};

#endif
