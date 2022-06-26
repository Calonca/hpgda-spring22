#pragma once
#include <vector>
#include "implementation.cuh"
#include <cuda_fp16.h>
#include <map>
#include "../personalized_pagerank.cuh"

class PersonalizedPageRank;
class FinalImplementation : public Implementation {
private:
    std::vector<float> prFloat;
    std::vector<float> valFloat;
    std::vector<double> danglingPpr;
    std::map<int, double> danglingPprTop19;
    std::vector<int> pDanglingIndexes;

    int* dangling_gpu;
    float* pDanglingFact_gpu;
    float* pSquareError_gpu;
    float squareError_cpu = INITIAL_SQUARE_ERROR;
    int danglingSize;
    float* pr_old;
    float* val_gpu;
    int* x_gpu;
    int* y_gpu;
    float* pr_gpu;
    int *pDanglingIndexes_gpu;
    int old_ppr_index = -1;
    const int THREADS = DEFAULT_BLOCK_SIZE;
    int reducedV;
    int BLOCKS_V;
    int BLOCKS_E;
    int BLOCKS_ERROR;
    int BLOCKS_D;
    float ppVertexConst;
    float dampingFract;
    int CUSTOM_MAX_ITERATIONS = 16;
    double* pr_gpu_double;

    const int BLOCK_SIZE = 256;
    const int MAX_BLOCKS = 40;
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    int num_units;
    int num_warps;
    int num_blocks;
    int num_iters;
    int interval_size;
    int tail; // do the last few nonzeros separately (fewer than WARP_SIZE elements)
    int active_warps;
    int *temp_rows;
    float *temp_vals;

public:
    void alloc();
    void init();
    void reset();
    void execute(int iter);
    void clean();
    bool isDangling(int vertex);
    void initDanglingPpr();
    void initDanglingPprTop19();
    void initDanglingIndexes();

};