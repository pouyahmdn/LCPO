//
// Created by Pouya Hamadanian on 8/9/22.
//

#include "iostream"
#include "WindowStatsPipe.h"
#include "cmath"
#include <numeric>
#include <algorithm>

float quick_percentile(std::vector<float> sorted_arr, float p){
    // 0 <= p <= 1
    if (p == 1)
        return sorted_arr.back();
    else {
        float index_p = p*float(sorted_arr.size()-1);
        int index_f = floor(index_p);
        return sorted_arr[index_f] + (sorted_arr[index_f+1] - sorted_arr[index_f]) * (index_p - (float)index_f);
    }
}

SingleStat<float> get_single_stat(std::vector<float> arr_data, bool sort){
    if (sort)
        std::sort(arr_data.begin(), arr_data.end());
    else if (arr_data.front() > arr_data.back()){
        std::cout << "Sorted argument is not really sorted, exiting..." << std::endl;
        exit(1);
    }
    return SingleStat<float>{
        .average=(float)std::accumulate(arr_data.begin(), arr_data.end(), 0.0) / (float)arr_data.size(),
        .minimum=arr_data.front(),
        .maximum=arr_data.back(),
        .median=quick_percentile(arr_data, 0.5),
        .per95=quick_percentile(arr_data, 0.95),
        .per97=quick_percentile(arr_data, 0.97),
        .per99=quick_percentile(arr_data, 0.99),
    };
}

void WindowStatsPipe::flush(){
    for (auto& pipe: next_pipes)
        pipe->flush();
}

void WindowStatsPipe::enqueue(void* entry){
    std::cout << "Called enqueue on stat calculating pipe, exiting..." << std::endl;
    exit(1);
}

void WindowStatsPipe::extend(std::vector<void*>& arr_entry){
    // Length
    auto* windowStat = new WindowStat{
        .inflation_avg=0,
        .model_avg=0,
        .tw_avg=0,
        .len=arr_entry.size(),
    };

    if (windowStat->len > 0){
        // Trace index
        windowStat->trace = ((LogEntry*)arr_entry.front())->trace_index;
        // Bucket duration
        windowStat->interval_duration = ((LogEntry*)arr_entry.back())->arrival - ((LogEntry*)arr_entry.front())->arrival;

        float data[9][windowStat->len];

        for (std::vector<void*>::size_type i = 0; i != arr_entry.size(); i++){
            auto* logEntry = (LogEntry*) arr_entry[i];

            data[0][i] = logEntry->arrival;
            data[1][i] = logEntry->delay;
            data[2][i] = logEntry->first_duration;
            data[3][i] = logEntry->duration;
            data[4][i] = logEntry->size;
            data[5][i] = logEntry->delay / logEntry->first_duration;
            data[6][i] = logEntry->delay / logEntry->duration;
            data[7][i] = logEntry->delay / logEntry->size;
            data[8][i] = logEntry->delay - logEntry->duration;

            windowStat->inflation_avg += logEntry->duration / logEntry->size;
            windowStat->model_avg += float(logEntry->model_index);
            windowStat->tw_avg += float(logEntry->tw);
        }
        // Inflation, Model index and TW average
        windowStat->inflation_avg /= float(windowStat->len);
        windowStat->model_avg /= float(windowStat->len);
        windowStat->tw_avg /= float(windowStat->len);

        // 9 Single stats
        windowStat->arrival = get_single_stat(std::vector<float>(data[0], data[0]+windowStat->len), false);
        windowStat->delay = get_single_stat(std::vector<float>(data[1], data[1]+windowStat->len), true);
        windowStat->proc_first = get_single_stat(std::vector<float>(data[2], data[2]+windowStat->len), true);
        windowStat->proc = get_single_stat(std::vector<float>(data[3], data[3]+windowStat->len), true);
        windowStat->size = get_single_stat(std::vector<float>(data[4], data[4]+windowStat->len), true);
        windowStat->delay_p1 = get_single_stat(std::vector<float>(data[5], data[5]+windowStat->len), true);
        windowStat->delay_proc = get_single_stat(std::vector<float>(data[6], data[6]+windowStat->len), true);
        windowStat->delay_size = get_single_stat(std::vector<float>(data[7], data[7]+windowStat->len), true);
        windowStat->qdelay = get_single_stat(std::vector<float>(data[8], data[8]+windowStat->len), true);
    }

    for (Pipe* pipe: next_pipes)
        pipe->enqueue((void*)windowStat);
    delete windowStat;
}