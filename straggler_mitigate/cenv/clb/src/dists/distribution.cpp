//
// Created by Pouya Hamadanian on 5/24/21.
//

#include "distribution.h"
#include <cassert>

// generating functions
double distribution::generate_bounded(std::mt19937 *_g, double bound_low, double bound_high) {
    assert(bound_low < bound_high);
    double ret_val = generate(_g);
    if (ret_val > bound_high)
        return bound_high;
    else if (ret_val < bound_low)
        return bound_low;
    return ret_val;
}
