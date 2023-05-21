//
// Created by Pouya Hamadanian on 5/24/21.
//

#ifndef CLB_DISTRIBUTION_H
#define CLB_DISTRIBUTION_H

//
// Created by Pouya Hamadanian on 5/24/21.
//

#include <random>

enum dist_type{
    normal, exponential, pareto, stat_ret
};

struct dist_param{
    enum dist_type type;
    double normal_mu, normal_sigma;
    double exponential_rev_lambda;
    double pareto_shape, pareto_scale;
    double stat_ret_val;
};

class distribution
{
public:
    virtual ~distribution() = default;;

    // generating functions
    virtual double generate(std::mt19937* _g) = 0;

    // average functions
    virtual double average() = 0;
    virtual double average(double bound_low, double bound_high) = 0;

    // generating functions
    double generate_bounded(std::mt19937* _g, double bound_low, double bound_high);
    virtual distribution* copy() = 0;
};

#endif
