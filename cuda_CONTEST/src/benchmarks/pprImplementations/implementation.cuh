//
// Created by calonca on 6/8/22.
//

#pragma once
#include <vector>

class PersonalizedPageRank;
class Implementation {
public:
    virtual void alloc();
    virtual void init();
    virtual void reset();
    virtual void execute(int iter);
    virtual void clean();
    PersonalizedPageRank* pPpr;
    bool debug;

};
