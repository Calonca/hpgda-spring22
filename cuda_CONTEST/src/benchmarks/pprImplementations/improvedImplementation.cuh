#pragma once
#include <vector>
#include "implementation.cuh"
#include "../personalized_pagerank.cuh"
class PersonalizedPageRank;
class ImprovedImplementation : public Implementation {
private:
    int* dangling_gpu;
    double* pDanglingFact_gpu;
    double* pSquareError_gpu;
    double* pr_old;
    double* pr_temp;
    double* val_gpu;
    int* x_gpu;
    int * xPtr_gpu;
    int* y_gpu;
    double* pr_gpu;
    unsigned int* count1;
    unsigned int* count2;

    const int THREADS = DEFAULT_BLOCK_SIZE;
    int BLOCKS_V;
    int BLOCKS_E;

    double* prMinus2;
    double* g;
    double* h;
    std::vector<int> xPtr; // for CSR, contains info about where a row starts and ends
public:
    void alloc();
    void init();
    void reset();
    void execute(int iter);
    void clean();
    void initCSR();

};