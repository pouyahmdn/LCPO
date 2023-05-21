//
// Created by Pouya Hamadanian on 8/9/22.
//

#include "FilePipe.h"

FilePipe::FilePipe(const std::string& file_address, long word_chunks) {
    this -> word_chunks = word_chunks;
    this -> active_handler = std::ofstream (file_address, std::ios::out | std::ios::binary);
}

void FilePipe::enqueue(void *entry) {
    active_handler.write(reinterpret_cast<const char *>(entry), word_chunks);
}

void FilePipe::flush(){
    active_handler.flush();
}

FilePipe::~FilePipe() {
    active_handler.flush();
    active_handler.close();
}
