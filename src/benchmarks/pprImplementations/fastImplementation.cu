//
// Created by calonca on 6/8/22.
//

#include "../personalized_pagerank.cuh"
#include "fastImplementation.cuh"

//////////////////////////////
//////////////////////////////

__global__ void incrementByValue(float *arr, int idx, float value) {
    arr[idx] += value;
}


__global__ void vectorScalarMul(const float scalar, float *vector, int array_len) {

    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] *= scalar;
        i += gridSize;
    }
}
__global__ void vectorScalarAdd(float scalar, float *vector, int array_len) {

    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] += scalar;
        i += gridSize;
    }
}

__global__ void init_vector(float* v, int size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        v[i] = value;
        i += gridDim.x * blockDim.x;
    }
}

__global__ void copy_vector(float* dest, int* source, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        dest[i] = source[i];
        i += gridDim.x * blockDim.x;
    }
}

//////////////////////////////
//////////////////////////////


void FastImplementation::dotCublas(const float *array1, const float *array2, float *result, int vector_len, float alpha,
                                   cublasHandle_t *handle) {

    cublasSdot (*handle, vector_len,
                         array1, 1,
                         array2, 1,
                         result);

    *result=alpha*(*result)/vector_len;
}

void FastImplementation::sqCublas(const float *old, float *newVector, float *result, int vector_len, cublasHandle_t *handle) {

    float A = -1.0;
    const float* pA = &A;

    cublasSaxpy(*handle, vector_len,
                         pA,
                         old, 1,
                         newVector, 1);
    cudaDeviceSynchronize();


    cublasSdot (*handle, vector_len,
                         newVector, 1,
                         newVector, 1,
                         result);

}


void FastImplementation::alloc() {
    // Load the input graph and preprocess it;
    pPpr->initialize_graph();

    cudaMallocManaged(&coo.x, sizeof(int) * pPpr->E);
    cudaMallocManaged(&csr.rowIndex, sizeof(int) * pPpr->E);
    cudaMallocManaged(&coo.y, sizeof(int) * pPpr->E);
    cudaMallocManaged(&coo.val, sizeof(float ) * pPpr->E);


    cudaMallocManaged(&dangling_gpu, sizeof(int) * pPpr->V);
    cudaMallocManaged(&pSquareError_gpu, sizeof(float ));

    cudaMallocManaged(&pr_gpu, sizeof(float)*pPpr->V);
    cudaMallocManaged(&pr_temp, sizeof(float)*pPpr->V);
    cudaMallocManaged(&pr_old, sizeof(float)*pPpr->V);

    cudaMallocManaged(&pDanglingFact_gpu, sizeof(float ));
    cudaMallocManaged(&pAlpha_gpu, sizeof(float ));
    cudaMallocManaged(&pTeleportFact_gpu, sizeof(float ));
    cudaMallocManaged(&pV_gpu, sizeof(int ));
    cudaMallocManaged(&pE_gpu, sizeof(int ));
    cudaMallocManaged(&pPersonalization_vertex_gpu, sizeof(int ));
}

//Print error if the block size is too small
void FastImplementation::checkBlockSize(int blockSize, int minBlockSize) {
    if (blockSize < minBlockSize && debug) {
        printf("The selected number of blocks is too small, %d will be used\n",minBlockSize);
    }
}

void FastImplementation::checkCuSparseStatus(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::string error_string = cusparseGetErrorString(status);
        printf("Cusparse error: %s\n", error_string.c_str());
    }
}


void FastImplementation::init() {
    cublasCreate(&cublasHandle);

    cusparseCreate(&cusparseHandle);

    valFloat.resize(pPpr->E);
    std::transform(pPpr->val.begin(), pPpr->val.end(), valFloat.begin(), [](double x) { return (float )x;});

    blocksVertex = 1+pPpr->V/pPpr->T;
    blocksEdge = 1+pPpr->E/pPpr->T;

    int blocks = std::max(blocksVertex,blocksEdge);

    checkBlockSize(pPpr->B,blocks);

    cudaMemcpy(coo.x,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice);
    cudaMemcpy(coo.y,pPpr->y.data(), sizeof(int) * pPpr->E,cudaMemcpyHostToDevice);
    cudaMemcpy(coo.val,valFloat.data(), sizeof(float ) * pPpr->E,cudaMemcpyHostToDevice);
    vectorScalarMul<<<blocksEdge, pPpr->T>>>(pPpr->alpha, coo.val, pPpr->E);
    cudaMemcpy(dangling_gpu, pPpr->dangling.data(), sizeof(int) * pPpr->V, cudaMemcpyHostToDevice);

    initCoo();
}

void FastImplementation::reset() {
    cudaMemset(pSquareError_gpu,INITIAL_SQUARE_ERROR, sizeof(float ));

    cudaMemset(pr_gpu,0.0, sizeof(float)*pPpr->V);
    cudaMemset(pr_temp,0.0, sizeof(float)*pPpr->V);
    cudaMemset(pr_old,0.0, sizeof(float)*pPpr->V);

    cudaMemset(pDanglingFact_gpu,0, sizeof(float ));
    cudaMemcpy(pAlpha_gpu,&(pPpr->alpha), sizeof(float ),cudaMemcpyHostToDevice);
    float tempTeleportFact = pPpr->alpha/pPpr->V;
    cudaMemcpy(pTeleportFact_gpu,&tempTeleportFact, sizeof(float ),cudaMemcpyHostToDevice);

    cudaMemcpy(pV_gpu,&(pPpr->V), sizeof(int ),cudaMemcpyHostToDevice);
    cudaMemcpy(pE_gpu,&(pPpr->E), sizeof(int ),cudaMemcpyHostToDevice);
    cudaMemcpy(pPersonalization_vertex_gpu,&(pPpr->personalization_vertex), sizeof(int ),cudaMemcpyHostToDevice);

    init_vector<<<blocksVertex,pPpr->T>>>(pr_old,pPpr->V,1.0f/float(pPpr->V));

}

void FastImplementation::execute(int iter) {

    if (debug) {
        printf("\nx initial value\n");
        print_gpu_array(coo.x, pPpr->E);
        printf("\ny initial value\n");
        print_gpu_array(coo.y, pPpr->E);
        printf("\nval initial value\n");
        print_gpu_array(coo.val, pPpr->E);
        printf("\npr_old initial value\n");
        print_gpu_array(pr_old, pPpr->V);
    }

    float squareError_cpu = INITIAL_SQUARE_ERROR;

    cudaMemset(pDanglingFact_gpu,0, sizeof(float ));
    cudaMemset(pr_temp,0, sizeof(float )*pPpr->V);
    cudaMemset(pSquareError_gpu,0, sizeof(float ));

    for (int i = 0; squareError_cpu > pPpr->convergence_threshold && i < pPpr->max_iterations; i++) {
        if (debug){
            printf("\nConvergence iteration %i pr_old\n",i);
            print_gpu_array(pr_old,pPpr->V);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_old,pPpr->V);}

        cudaMemset(pDanglingFact_gpu,0, sizeof(float ));
        cudaMemset(pr_temp,0, sizeof(float )*pPpr->V);
        cudaMemset(pSquareError_gpu,0, sizeof(float ));

        thrust::device_vector<float> danglingFloat(pPpr->V);
        copy_vector<<<blocksVertex,pPpr->T>>>(thrust::raw_pointer_cast(danglingFloat.data()),dangling_gpu,pPpr->V);
        dotCublas(thrust::raw_pointer_cast(danglingFloat.data()), pr_old, pDanglingFact_gpu, *pV_gpu, pPpr->alpha,
                  &cublasHandle);
        cudaDeviceSynchronize();

        if (debug)
            printf("\nIteration %i dangling is: %f \n",i,*pDanglingFact_gpu);


        float alpha = 1.0f;
        float beta = 0.0f;
        size_t bufferSize = pPpr->V * sizeof(float);

        checkCuSparseStatus(cusparseDnVecSetValues(pr_old_descr,pr_old));
        //cusparseDnVecGetValues(pr_old_descr, reinterpret_cast<void **>(&pr_temp));
        cudaDeviceSynchronize();


        checkCuSparseStatus(cusparseSpMV_bufferSize(cusparseHandle,
                                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha,
                                                    coo.descr,
                                                    pr_old_descr,
                                                    &beta,
                                                    pr_temp_descr,
                                                    CUDA_R_32F,
                                                    coo.alg,
                                                    &bufferSize));

        cudaDeviceSynchronize();

        void *buffer;
        if (bufferSize>0) {
            cudaMalloc(&buffer, bufferSize);
            std::cout << "\nBuffer size is: " << bufferSize << std::endl;
        }

        checkCuSparseStatus(cusparseSpMV(cusparseHandle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         coo.descr,
                                         pr_old_descr,
                                         &beta,
                                         pr_temp_descr,
                                         CUDA_R_32F,
                                         coo.alg,
                                         &buffer));

        if (bufferSize>0)
            cudaFree(buffer);
        checkCuSparseStatus(cusparseDnVecGetValues(pr_temp_descr, reinterpret_cast<void **>(&pr_temp)));
        cudaDeviceSynchronize();

        if (debug) {
            printf("\npr_temp after coo\n");
            print_gpu_array(pr_temp, pPpr->V);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_temp,pPpr->V);
        }

        vectorScalarAdd<<<blocksVertex,pPpr->T>>>(*pDanglingFact_gpu,pr_temp,*pV_gpu);
        cudaDeviceSynchronize();

        if(debug) {
            printf("\npr_temp after adding dangling\n");
            print_gpu_array(pr_temp, pPpr->V);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_temp, pPpr->V);
        }

        incrementByValue<<<1, 1>>>(pr_temp, pPpr->personalization_vertex, 1.0 - pPpr->alpha);

        cudaMemcpy(pr_gpu,pr_temp ,sizeof(float)*pPpr->V,cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();

        if (debug) {
            printf("\npr_gpu after increment of 1-alpha in position %d\n", pPpr->personalization_vertex);
            print_gpu_array(pr_gpu, 20);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_gpu, pPpr->V);
        }

        sqCublas(pr_old, pr_temp,pSquareError_gpu, pPpr->V,&cublasHandle);

        cudaMemcpy(&squareError_cpu,pSquareError_gpu,sizeof (float ),cudaMemcpyDeviceToHost);
        squareError_cpu= std::sqrt(squareError_cpu);
        cudaDeviceSynchronize();

        if (debug)
        {
            printf("\nIteration %i Square error is: %f \n",i,squareError_cpu);
        }
        cudaMemcpy(pr_old,pr_gpu ,sizeof(float)*pPpr->V,cudaMemcpyDeviceToDevice);
    }

    prFloat.resize(pPpr->V);
    cudaMemcpy(prFloat.data(),pr_gpu,sizeof (float )*pPpr->V,cudaMemcpyDeviceToHost);
    std::transform(prFloat.begin(), prFloat.end(), pPpr->pr.begin(), [](float x) { return (double )x;});
}

void FastImplementation::clean() {
    cudaFree(coo.x);
    cudaFree(coo.y);
    cudaFree(coo.val);

    cudaFree(dangling_gpu);
    cudaFree(pSquareError_gpu);

    cudaFree(pr_gpu);
    cudaFree(pr_temp);
    cudaFree(pr_old);


    cudaFree(pDanglingFact_gpu);
    cudaFree(pAlpha_gpu);
    cudaFree(pTeleportFact_gpu);
    cudaFree(pV_gpu);
    cudaFree(pE_gpu);
    cudaFree(pPersonalization_vertex_gpu);


    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);

}

void FastImplementation::initCoo() {
    checkCuSparseStatus(cusparseCreateDnVec(&pr_temp_descr,pPpr->V,pr_temp,CUDA_R_32F));
    checkCuSparseStatus(cusparseCreateDnVec(&pr_old_descr,pPpr->V,pr_old,CUDA_R_32F));


    checkCuSparseStatus(cusparseCreateCoo(
            &coo.descr,
            pPpr->V,
            pPpr->V,
            pPpr->E,
            coo.x,
            coo.y,
            coo.val,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));

}

