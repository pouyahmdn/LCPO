//
// Created by Pouya Hamadanian on 5/24/21.
//

#include <iostream>
#include "normal_dist.h"
#include "cassert"
#include <limits>

double normal_dist::generate(std::mt19937 *_g) {
    double z;
    if (prev) {
        prev = false;
        z = prev_z;
    } else {
        double u, v, r;
        do {
            u = std::generate_canonical<double, std::numeric_limits<double>::digits>(*_g) * 2 - 1;
            v = std::generate_canonical<double, std::numeric_limits<double>::digits>(*_g) * 2 - 1;
            r = u * u + v * v;
        } while (r > 1 || r == 0);
        r = sqrt(-2 * log(r) / r);
        prev_z = v * r;
        prev = true;
        z = u * r;
    }
    return z * _sigma_ + _mu_;
}

double normal_dist::average(double bound_low, double bound_high) {
    assert(bound_low < bound_high);
    double lowPhi, low_phi, highPhi, high_phi;

    if (bound_low > -std::numeric_limits<double>::infinity()) {
        double cen = (bound_low - _mu_) / _sigma_;
        lowPhi = (0.5 + 0.5 * erf(cen * M_SQRT1_2));
        low_phi = 0.5 * M_2_SQRTPI * M_SQRT1_2 * exp(-cen * cen / 2);
    } else {
        lowPhi = 0;
        low_phi = 0;
    }
    if (bound_high < std::numeric_limits<double>::infinity()) {
        double cen = (bound_high - _mu_) / _sigma_;
        highPhi = (0.5 + 0.5 * erf(cen * M_SQRT1_2));
        high_phi = 0.5 * M_2_SQRTPI * M_SQRT1_2 * exp(-cen * cen / 2);
    } else {
        highPhi = 1;
        high_phi = 1;
    }
    return _mu_ + (low_phi - high_phi) / (highPhi - lowPhi) * _sigma_;
}

double normal_dist::average() {
    return _mu_;
}

normal_dist *normal_dist::copy() {
    return new normal_dist(*this);
}
