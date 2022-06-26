#pragma once
#include "implementation.cuh"
#include "cublas_v2.h"
#include "cusparse_v2.h"


class PersonalizedPageRank;
class CublasCusparseNaiveImplementation : public Implementation {
private:
    struct cooMatrix
    {
        cusparseSpMatDescr_t descr;
        int* x_gpu;
        cusparseDnVecDescr_t x_descr;
        int* y_gpu;
        float* val_gpu;
        cusparseSpMVAlg_t alg = CUSPARSE_COOMV_ALG;
    };
    struct bsrMatrix
    {
        const int bsrBlockDim = 4;
        int blocksInMat;
        int nnZBlocks;
        cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
        cusparseMatDescr_t bsr_desc;
        int* rowPtrs;
        int* colIdxs;
        float* val;
    };

    struct csrMat
    {
        cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG1;
        cusparseMatDescr_t mat_desc;
        int* rowIndex;
    };

    std::vector<float> prFloat;
    std::vector<float> valFloat;

    int blocksVertex;
    int blocksEdge;

    float* pAlpha_gpu;
    float* pTeleportFact_gpu;
    int* pE_gpu;
    int* pPersonalization_vertex_gpu;
    int* dangling_gpu;
    float* pDanglingFact_gpu;
    float* pSquareError_gpu;
    int* pV_gpu;
    float* pr_old;
    float* pr_temp;
    float* pr_gpu;
    cooMatrix coo;
    csrMat csr;
    bsrMatrix bsr;

    cublasHandle_t cublasHandle;

    cusparseDnVecDescr_t pr_old_descr;
    cusparseDnVecDescr_t pr_temp_descr;

    //BSR format

    float* prBrs;
    float* prTempBrs;

public:
    void dotCublas(const float *array1, const float *array2, float *result, int vector_len, float alpha, cublasHandle_t* handle);
    void sqCublas(const float *old,float *newVector, float* result, int vector_len,cublasHandle_t* handle);
    void checkCuSparseStatus(cusparseStatus_t status);
    void checkBlockSize(int blockSize,int minBlockSize);
    void cooToBsr();
    void alloc();
    void init();
    void reset();
    void execute(int iter);
    void clean();

    cusparseHandle_t cusparseHandle;
};