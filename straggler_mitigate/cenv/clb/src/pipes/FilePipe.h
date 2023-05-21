//
// Created by Pouya Hamadanian on 8/9/22.
//

#ifndef CLB_FILEPIPE_H
#define CLB_FILEPIPE_H

#include "pipe.h"
#include <fstream>
#include <string>

class FilePipe: public Pipe {
private:
    std::ofstream active_handler;
    long word_chunks;

public:
    FilePipe(const std::string& file_address, long word_chunks);
    ~FilePipe() override;

    void flush() override;
    void enqueue(void* entry) override;
};

#endif //CLB_FILEPIPE_H
