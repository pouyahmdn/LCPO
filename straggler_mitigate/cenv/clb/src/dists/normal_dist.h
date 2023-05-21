//
// Created by Pouya Hamadanian on 5/24/21.
//

#ifndef CLB_NORMAL_DIST_H
#define CLB_NORMAL_DIST_H

#include<random>
#include<cmath>
#include "distribution.h"

class normal_dist : public distribution
{
private:
    const double _mu_;
    const double _sigma_;
    double prev_z{};
    bool prev;

public:
    // constructors and reset functions
    explicit normal_dist(const double mu, const double sigma) : _mu_(mu), _sigma_(sigma), prev(false) {}

    // generating functions
    double generate(std::mt19937* _g) override;

    // average functions
    double average() override;
    double average(double bound_low, double bound_high) override;
    normal_dist* copy() override;
};

#endif
