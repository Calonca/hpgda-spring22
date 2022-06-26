#pragma once
#include "implementation.cuh"
#include "curand_kernel.h"

struct cooEl{
    int x;
    //int idx;
    int y;
    //float pr;

    bool operator<( cooEl const& other ) const noexcept
    {
        return y < other.y;
    }
};


struct cooMatrix
{
    //On GPU
    int* x;
    //On cpu
    //float* val;
    std::map<int,std::vector<int>> adjList;
    std::vector<cooEl> elms;
};

struct cscMat
{
    //On GPU
    int* x;
    int* xPtr;
    int* neightSize;
    //On cpu
    std::vector<int> x_cpu;
    std::vector<int> neightSize_cpu;
    std::vector<int> xPtr_cpu;
};

class PersonalizedPageRank;
class MCCompletePath : public Implementation {
private:
    std::vector<float> prFloat;

    float* pr_gpu;
    float* initialPr_gpu;
    cooMatrix coo;
    cscMat csc;

    curandState* states;
    curandStatePhilox4_32_10_t* statesPhilox;
    curandStateMRG32k3a_t* statesMRG;
    int maxWalkLen;
    int walkers;
    float stopThreshold;
    int spawnWalkerThreshold;


public:
    void alloc();
    void init();
    void initQuasiRandom();
    void initPr(bool initGPU, bool initCPU);
    void reset();
    void execute(int iter);
    void clean();
    void sortAndConvert();
    void cooToCsc();
    void getKClosest(int k, std::vector<int> &closest, int pVertex);
    void MCCompletePathAlgoCPU(int s, int nWalkers);
};