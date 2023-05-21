//
// Created by Pouya Hamadanian on 5/20/21.
//

#include "ActionSpace.h"

ActionSpace::ActionSpace(unsigned short n, const short* timeouts) {
    this -> n = n;
    this -> timeouts = new short [n];
    for (int i = 0; i < n; i++)
        this -> timeouts[i] = timeouts[i];
}

ActionSpace::ActionSpace(ActionSpace *act_space_base) {
    this->n = act_space_base->n;
    this -> timeouts = new short [this->n];
    for (int i = 0; i < this->n; i++)
        this -> timeouts[i] = act_space_base->timeouts[i];

}

bool ActionSpace::contains(unsigned short action) const {
    if (action < n)
        return true;
    return false;
}

ActionSpace::~ActionSpace() {
    delete timeouts;
}


