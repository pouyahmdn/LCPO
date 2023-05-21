//
// Created by Pouya Hamadanian on 5/24/21.
//

#ifndef CLB_PARETO_DIST_H
#define CLB_PARETO_DIST_H

#include "distribution.h"

class pareto_dist : public distribution {
private:
    const double _shape_;
    const double _scale_;

public:
    // constructors and reset functions
    explicit pareto_dist(double shape = 1, double scale = 1) : _shape_(shape), _scale_(scale) {}

    // generating functions
    double generate(std::mt19937* _g) override;

    // average functions
    double average() override;
    double average(double bound_low, double bound_high) override;
    pareto_dist* copy() override;
};


#endif
