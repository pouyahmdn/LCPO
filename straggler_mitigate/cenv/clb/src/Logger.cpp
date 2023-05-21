//
// Created by Pouya Hamadanian on 5/19/21.
//

// COLUMNS = ["ID", "TIMEOUT", "FIRST", "ARRIVAL", "DELAY", "NUM_INSTANCES", "INSTANCE_INDEX",
// "QUEUE", "QUEUE_FIRST", "SIZE", "MODEL_INDEX", "TRAINING WHEELS", "TRACE_INDEX", "PROC", "PROC_FIRST"]
// 'ibhbffbbbbfb?bff'     i:4   b:1   h:2   f:4   ?:1   total: 33

#include <iostream>
#include "Logger.h"
#include "cassert"

Logger::Logger(double split_size, const std::string& file_address, bool skip_log) {
    this -> skip = skip_log;
    if (!this -> skip){
        this -> size_so_far = 0;
        this -> split_size = split_size;
        this -> file_address = file_address;
        this -> index_file = 0;
        this -> active_handler = std::ofstream (file_address + "_0", std::ios::out | std::ios::binary);
        assert(sizeof(unsigned int) == 4);
        assert(sizeof(unsigned char) == 1);
        assert(sizeof(bool) == 1);
        assert(sizeof(float) == 4);
    }
}

void Logger::log_job(Job *job) {
    if (!skip){
        active_handler.write(reinterpret_cast<const char *>(&(job -> id)), sizeof(unsigned int));
        active_handler.write(reinterpret_cast<const char *>(&(job -> timeout_idx)), sizeof(unsigned char));
        active_handler.write(reinterpret_cast<const char *>(&(job -> first_completed)), sizeof(bool));
        float arrival = (float) job -> arrival_time;
        active_handler.write(reinterpret_cast<const char *>(&(arrival)), sizeof(float));
        float delay = (float) job -> get_delay();
        active_handler.write(reinterpret_cast<const char *>(&(delay)), sizeof(float));
        active_handler.write(reinterpret_cast<const char *>(job -> num_instances), sizeof(unsigned char));
        active_handler.write(reinterpret_cast<const char *>(&(job -> instance_index)), sizeof(unsigned char));
        active_handler.write(reinterpret_cast<const char *>(&(job -> queue_observed)), sizeof(unsigned char));
        unsigned char first_queue_obs = job -> get_first_queue_obs();
        active_handler.write(reinterpret_cast<const char *>(&(first_queue_obs)), sizeof(unsigned char));
        float orig_duration = (float) job -> original_duration;
        active_handler.write(reinterpret_cast<const char *>(&(orig_duration)), sizeof(float));
        active_handler.write(reinterpret_cast<const char *>(&(job -> model_index)), sizeof(unsigned char));
        active_handler.write(reinterpret_cast<const char *>(&(job -> tw_driven)), sizeof(bool));
        active_handler.write(reinterpret_cast<const char *>(&(job -> trace_origin_index)), sizeof(unsigned char));
        float duration = (float) job -> get_duration();
        active_handler.write(reinterpret_cast<const char *>(&(duration)), sizeof(float));
        float first_duration = (float) job -> get_first_duration();
        active_handler.write(reinterpret_cast<const char *>(&(first_duration)), sizeof(float));
        size_so_far += 33;
        if (size_so_far > split_size){
            size_so_far -= split_size;
            index_file += 1;
            active_handler.flush();
            active_handler.close();
            active_handler = std::ofstream (file_address + "_" + std::to_string(index_file), std::ios::out | std::ios::binary);
        }
    }
}

void Logger::flush(){
    if (!skip)
        active_handler.flush();
}

Logger::~Logger() {
    if (!skip){
        active_handler.flush();
        active_handler.close();
    }
}
