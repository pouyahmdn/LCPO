//
// Created by Pouya Hamadanian on 5/20/21.
//

#include <cmath>
#include "utils.h"
#include <limits>
#include <cassert>

double avg_partition_multi_array(double **data, const int *length_data, int num_rows,
                                 double partition_ratio) {
    int total_length = 0;
    double pivot;
    for (int i = 0; i < num_rows; i++) {
        total_length += length_data[i];
        if (length_data[i] > 0)
            pivot = data[i][0];
    }
    assert(total_length != 0);
    auto *data_1dim = new double[total_length];
    int index_pre = 0;
    int index_post = total_length - 1;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < length_data[i]; j++) {
            if (data[i][j] > pivot)
                data_1dim[index_pre++] = data[i][j];
            else
                data_1dim[index_post--] = data[i][j];
        }
    }
    double n_th = partition_ratio * total_length;
    // index_pre now points to the start of post array and index_post points to the end of pre array
    // so index_pre = index_post + 1
    double ret_val;
    if (n_th < index_pre)
        ret_val = sum_partition(data_1dim, index_pre, n_th);
    else if (n_th < index_pre + 1)
        ret_val = sum(data_1dim, index_pre) + (n_th - index_pre) * max(data_1dim + index_pre, total_length - index_pre);
    else
        ret_val = sum(data_1dim, index_pre) + sum_partition(data_1dim + index_pre, total_length - index_pre, n_th - index_pre);
    delete[] data_1dim;
    return ret_val / n_th;
}

double sum_partition(double *data, int length, double n_th) {
    if (length == 2) {
        if (data[0] < data[1]){
            double temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
        if (n_th <= 1)
            return data[0] * n_th;
        else
            return data[0] + data[1] * (n_th-1);
    }
    if (length == 1) {
        return data[0] * n_th;
    }
    int index_pre = -1;
    int index_post = length;
    double pivot = data[length / 2];
    while (true) {
        do {
            index_pre += 1;
        } while (index_pre < length and data[index_pre] > pivot);
        do {
            index_post -= 1;
        } while (index_post >= 0 and data[index_post] < pivot);
        if (index_pre >= index_post)
            break;
        double temp = data[index_pre];
        data[index_pre] = data[index_post];
        data[index_post] = temp;
    }
    index_pre = index_post + 1;

    if (n_th < index_pre)
        return sum_partition(data, index_pre, n_th);
    else if (n_th < index_pre + 1)
        return sum(data, index_pre) + (n_th - index_pre) * max(data + index_pre, length - index_pre);
    else
        return sum(data, index_pre) + sum_partition(data + index_pre, length - index_pre, n_th - index_pre);
}

double sum(const double *data, unsigned int length) {
    double sum = 0;
    for (unsigned int i = 0; i < length; i++)
        sum += data[i];
    return sum;
}

int sum(const int *data, unsigned int length) {
    int sum = 0;
    for (unsigned int i = 0; i < length; i++)
        sum += data[i];
    return sum;
}

double average(const double *data, unsigned int length) {
    return sum(data, length) / length;
}

double max(const double *data, unsigned int length) {
    double most = -std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < length; i++)
        if (most < data[i])
            most = data[i];
    return most;
}

double min(const double *data, unsigned int length) {
    double least = std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < length; i++)
        if (least > data[i])
            least = data[i];
    return least;
}

double percentile(double *data, int length, double percentile) {
    if (length == 1)
        return data[0];
    if (length == 2){
        if (data[0] <= data[1])
            return data[0] * (1 - percentile) + data[1] * percentile;
        else
            return data[1] * (1 - percentile) + data[0] * percentile;
    }
    auto *target = new double[length];
    int index_pre = 0;
    int index_post = length-1;
    double pivot = data[length / 2];
    for (int j = 0; j < length; j++) {
        if (data[j] < pivot)
            target[index_pre++] = data[j];
        else
            target[index_post--] = data[j];
    }
    double n_th = percentile * (length - 1);
    // index_pre now points to the start of post array and index_post points to the end of pre array
    // so index_pre = index_post + 1
    double ret_val;
    if (n_th < index_post)
        ret_val = percentile_pivot(target, index_pre, n_th);
    else if (n_th > index_pre)
        ret_val = percentile_pivot(target + index_pre, length - index_pre, n_th - index_pre);
    else
        ret_val = max(target, index_pre) * (index_pre - n_th) +
                  min(target + index_pre, length - index_pre) * (n_th - index_post);
    delete[] target;
    return ret_val;
}

double percentile_pivot(double *data, int length, double n_th) {
    if (length == 2) {
        if (data[0] <= data[1])
            return data[0] * (1 - n_th) + data[1] * n_th;
        else
            return data[1] * (1 - n_th) + data[0] * n_th;
    }

    int index_pre = -1;
    int index_post = length;
    double pivot = data[length / 2];
    while (true) {
        do {
            index_pre += 1;
        } while (index_pre < length and data[index_pre] < pivot);
        do {
            index_post -= 1;
        } while (index_post >= 0 and data[index_post] > pivot);
        if (index_pre >= index_post)
            break;
        double temp = data[index_pre];
        data[index_pre] = data[index_post];
        data[index_post] = temp;
    }
    index_pre = index_post + 1;

    // index_pre now points to the start of post array and index_post points to the end of pre array
    // so index_pre = index_post + 1
    if (n_th < index_post)
        return percentile_pivot(data, index_pre, n_th);
    else if (n_th > index_pre)
        return percentile_pivot(data + index_pre, length - index_pre, n_th - index_pre);
    else
        return max(data, index_pre) * (index_pre - n_th) +
               min(data + index_pre, length - index_pre) * (n_th - index_post);
}

int binary_search_right_side(double* sorted_data, int length, double value){
    if (length == 1)
        return 0;
    int mid_index = length / 2;
    if (value < sorted_data[mid_index])
        return binary_search_right_side(sorted_data, mid_index, value);
    else
        return mid_index+binary_search_right_side(sorted_data+mid_index, length-mid_index, value);
}
