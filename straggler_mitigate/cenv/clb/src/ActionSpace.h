//
// Created by Pouya Hamadanian on 5/20/21.
//

#ifndef CLB_ACTIONSPACE_H
#define CLB_ACTIONSPACE_H


class ActionSpace {
public:
    unsigned short n;
    short* timeouts;

    ActionSpace(unsigned short n, const short* timeouts);
    ActionSpace(ActionSpace* act_space_base);
    ~ActionSpace();
    bool contains(unsigned short action) const;

};


#endif
