//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_LOGGER_H
#define CLB_LOGGER_H

#include <fstream>
#include <string>
#include "Job.h"

class Logger {
private:
    double split_size;
    unsigned short index_file;
    double size_so_far;
    std::string file_address;
    std::ofstream active_handler;
    bool skip;

public:
    Logger(double split_period, const std::string& file_address, bool skip_log);
    void log_job(Job* job);
    void flush();
    ~Logger();
};


#endif //CLB_LOGGER_H
