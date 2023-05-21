//
// Created by Pouya Hamadanian on 5/19/21.
//

#include <queue>
#include "LogSort.h"
#include "stdexcept"

LogSorter::LogSorter(const std::string& file_out, bool valid, unsigned int size_heap) {
    this -> pq = new std::priority_queue<LogEntry*, std::vector<LogEntry*>, CompareLogEntry>;
    this -> valid = valid;
    this -> active_handler = std::ofstream (file_out, std::ios::out | std::ios::binary);
    this -> size_heap = size_heap;
}

void LogSorter::add_file(const std::string& file_in){
    std::ifstream in_handle = std::ifstream (file_in, std::ios::in | std::ios::binary);

    std::streampos fsize = in_handle.tellg();
    in_handle.seekg(0, std::ios::end);
    fsize = in_handle.tellg() - fsize;
    if (fsize % 33 != 0)
        throw std::length_error("File size not a multiple of 61");

    in_handle.seekg(0, std::ios_base::beg);
    char* buffer = new char [size_entry];
    while (in_handle.peek() != EOF){
        in_handle.read(buffer, size_entry);
        if (is_first(buffer) == valid){
            LogEntry* newEntry = translateEntry(buffer);
            pq -> push(newEntry);
            while (pq -> size() >= size_heap)
                pop_and_write();
        }
    }
    in_handle.close();
    delete[] buffer;
}

LogEntry* LogSorter::translateEntry(const char* buff){
    return new LogEntry{
            .id=*(unsigned int *)(buff),
            .timeout_idx=*(unsigned char *)(buff+sizeof(unsigned int)),
            .first=*(bool *)(buff+sizeof(unsigned int)+sizeof(unsigned char)),
            .arrival=*(float *)(buff+sizeof(unsigned int)+sizeof(unsigned char)+sizeof(bool)),
            .delay=*(float *)(buff+sizeof(unsigned int)+sizeof(unsigned char)+sizeof(bool)+sizeof(float)),
            .num_instances=*(unsigned char *)(buff+sizeof(unsigned int)+sizeof(unsigned char)+sizeof(bool)+2*sizeof(float)),
            .instance_index=*(unsigned char *)(buff+sizeof(unsigned int)+2*sizeof(unsigned char)+sizeof(bool)+2*sizeof(float)),
            .queue_obs=*(unsigned char *)(buff+sizeof(unsigned int)+3*sizeof(unsigned char)+sizeof(bool)+2*sizeof(float)),
            .queue_obs_first=*(unsigned char *)(buff+sizeof(unsigned int)+4*sizeof(unsigned char)+sizeof(bool)+2*sizeof(float)),
            .size=*(float *)(buff+sizeof(unsigned int)+5*sizeof(unsigned char)+sizeof(bool)+2*sizeof(float)),
            .model_index=*(unsigned char *)(buff+sizeof(unsigned int)+5*sizeof(unsigned char)+sizeof(bool)+3*sizeof(float)),
            .tw=*(bool *)(buff+sizeof(unsigned int)+6*sizeof(unsigned char)+sizeof(bool)+3*sizeof(float)),
            .trace_index=*(unsigned char *)(buff+sizeof(unsigned int)+6*sizeof(unsigned char)+2*sizeof(bool)+3*sizeof(float)),
            .duration=*(float *)(buff+sizeof(unsigned int)+7*sizeof(unsigned char)+2*sizeof(bool)+3*sizeof(float)),
            .first_duration=*(float *)(buff+sizeof(unsigned int)+7*sizeof(unsigned char)+2*sizeof(bool)+4*sizeof(float)),
    };
}

bool LogSorter::is_first(const char* buff){
    return *(bool *)(buff+sizeof(unsigned int)+sizeof(unsigned char));
}

void LogSorter::writeEntry(LogEntry* entry){
    active_handler.write(reinterpret_cast<const char *>(&(entry -> id)), sizeof(unsigned int));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> timeout_idx)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> first)), sizeof(bool));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> arrival)), sizeof(float));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> delay)), sizeof(float));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> num_instances)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> instance_index)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> queue_obs)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> queue_obs_first)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> size)), sizeof(float));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> model_index)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> tw)), sizeof(bool));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> trace_index)), sizeof(unsigned char));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> duration)), sizeof(float));
    active_handler.write(reinterpret_cast<const char *>(&(entry -> first_duration)), sizeof(float));
}

void LogSorter::pop_and_write() {
    LogEntry* ret_pop = pq -> top();
    pq -> pop();
    writeEntry(ret_pop);
    delete ret_pop;
}

void LogSorter::flush(){
    while(!pq -> empty())
        pop_and_write();
}

LogSorter::~LogSorter() {
    delete pq;
    active_handler.flush();
    active_handler.close();
}
