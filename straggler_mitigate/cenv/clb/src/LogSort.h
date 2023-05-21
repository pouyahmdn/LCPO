//
// Created by Pouya Hamadanian on 5/19/21.
//

#ifndef CLB_LOGSORT_H
#define CLB_LOGSORT_H

#include <fstream>
#include "queue"
#include <string>

struct LogEntry{
    unsigned int id;
    unsigned char timeout_idx;
    bool first;
    float arrival;
    float delay;
    unsigned char num_instances;
    unsigned char instance_index;
    unsigned char queue_obs;
    unsigned char queue_obs_first;
    float size;
    unsigned char model_index;
    bool tw;
    unsigned char trace_index;
    float duration;
    float first_duration;
};

struct CompareLogEntry {
    bool operator()(const LogEntry* lhs, const LogEntry* rhs) const{
        return lhs -> id > rhs -> id;
    }
};

class LogSorter {
private:
    std::priority_queue<LogEntry*, std::vector<LogEntry*>, CompareLogEntry>* pq;
    std::ofstream active_handler;
    unsigned int size_heap;
    bool valid;
    const int size_entry = sizeof(unsigned int)+7*sizeof(unsigned char)+2*sizeof(bool)+5*sizeof(float);


    static LogEntry* translateEntry(const char* buff);
    void writeEntry(LogEntry* entry);
    void pop_and_write();
    static bool is_first(const char *buff);

public:

    LogSorter(const std::string& file_out, bool valid, unsigned int size_heap);
    ~LogSorter();
    void add_file(const std::string& file_in);
    void flush();
};


#endif
