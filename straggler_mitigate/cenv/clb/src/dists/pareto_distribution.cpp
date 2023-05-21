//
// Created by Pouya Hamadanian on 5/24/21.
//

#include<random>
#include<cmath>
#include "pareto_distribution.h"
#include "cassert"
#include <limits>

double pareto_dist::generate(std::mt19937 *_g) {
    return 1 / pow(1 - std::generate_canonical<double, std::numeric_limits<double>::digits>(*_g), 1 / _shape_) * _scale_;
}

double pareto_dist::average() {
    return _shape_/(1-_shape_) * _scale_;
}

double pareto_dist::average(double bound_low, double bound_high) {
    assert(bound_low < bound_high);
    double high_term, low_term;
    if (bound_low <= _scale_)
        low_term = _scale_ * _shape_ / (_shape_ - 1);
    else
        low_term = _scale_ * _shape_ / (_shape_ - 1) * pow(_scale_/bound_low, _shape_-1) - bound_low * (1 -  pow(_scale_/bound_low, _shape_));
    if (bound_high <= 0)
        high_term = _scale_;
    else if (bound_high < std::numeric_limits<double>::infinity())
        high_term = _scale_ * _shape_ / (_shape_ - 1) * pow(_scale_/bound_high, _shape_-1) - bound_high * pow(_scale_/bound_high, _shape_);
    else{
        high_term = 0;
    }
    return low_term-high_term;
}

pareto_dist *pareto_dist::copy() {
    return new pareto_dist(*this);
}
