#pragma once
#include "implementation.cuh"
#include "../personalized_pagerank.cuh"
class PersonalizedPageRank;
class NaiveImplementation : public Implementation {
private:
    double* pAlpha_gpu;
    double* pTeleportFact_gpu;
    int* pE_gpu;
    int* pPersonalization_vertex_gpu;
    int* dangling_gpu;
    double* pDanglingFact_gpu;
    double* pSquareError_gpu;
    int* pV_gpu;
    double* pr_old;
    double* pr_temp;
    double* val_gpu;
    int* x_gpu;
    int* y_gpu;
    double* pr_gpu;
    const int THREADS = DEFAULT_BLOCK_SIZE;
    int BLOCKS_V;
    int BLOCKS_E;
public:
    void alloc();
    void init();
    void reset();
    void execute(int iter);
    void clean();
};
