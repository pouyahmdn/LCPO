//
// Created by Pouya Hamadanian on 5/24/21.
//

#include<random>
#include<cmath>
#include "exponential_distribution.h"
#include "cassert"
#include <limits>

double exponential_dist::generate(std::mt19937* _g)
{
    return -log(1 - std::generate_canonical<double, std::numeric_limits<double>::digits>(*_g)) * _rev_lambda_;
}

double exponential_dist::average() {
    return _rev_lambda_;
}

double exponential_dist::average(double bound_low, double bound_high) {
    assert(bound_low < bound_high);
    double high_term, low_term;
    if (bound_low <= 0)
        low_term =  _rev_lambda_;
    else
        low_term = _rev_lambda_ * exp(-bound_low/_rev_lambda_);
    if (bound_high <= 0)
        high_term = _rev_lambda_;
    else if (bound_high < std::numeric_limits<double>::infinity())
        high_term = _rev_lambda_ * exp(-bound_high/_rev_lambda_);
    else{
        high_term = 0;
    }
    return low_term-high_term;
}

exponential_dist *exponential_dist::copy() {
    return new exponential_dist(*this);
}
