//
// Created by Pouya Hamadanian on 5/19/21.
//

#include "WallTime.h"

WallTime::WallTime() {
    this -> curr_time = 0;
}

WallTime::WallTime(WallTime* time_base) {
    this -> curr_time = time_base -> curr_time;
}

WallTime::~WallTime() = default;

void WallTime::update(double new_time) {
    curr_time = new_time;
}
void WallTime::reset() {
    curr_time = 0;
}