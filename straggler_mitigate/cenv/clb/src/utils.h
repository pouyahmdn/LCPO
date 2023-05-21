//
// Created by Pouya Hamadanian on 5/20/21.
//

#ifndef CLB_UTILS_H
#define CLB_UTILS_H

double avg_partition_multi_array(double **data, const int *length_data, int num_rows,
                                 double partition_ratio);

double sum_partition(double* data, int length, double n_th);

double average(const double *data, unsigned int length);

double sum(const double *data, unsigned int length);

int sum(const int *data, unsigned int length);

double max(const double *data, unsigned int length);

double min(const double *data, unsigned int length);

double percentile(double* data, int length, double percentile);

double percentile_pivot(double* data, int length, double percentile);

int binary_search_right_side(double* sorted_data, int length, double value);

#endif
