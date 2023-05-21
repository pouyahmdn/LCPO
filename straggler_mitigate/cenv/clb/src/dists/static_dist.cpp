//
// Created by Pouya Hamadanian on 5/24/21.
//

#include "static_dist.h"

double static_dist::generate(std::mt19937 *_g) {
    return this -> _ret_val_;
}

double static_dist::average() {
    return _ret_val_;
}

double static_dist::average(double bound_low, double bound_high) {
    if (_ret_val_ > bound_high)
        return bound_high;
    else if (_ret_val_ < bound_low)
        return bound_low;
    return _ret_val_;
}

static_dist *static_dist::copy() {
    return new static_dist(*this);
}
