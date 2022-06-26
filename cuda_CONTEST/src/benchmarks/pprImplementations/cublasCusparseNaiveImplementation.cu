//
// Created by calonca on 6/8/22.
//

#include "../personalized_pagerank.cuh"
#include "cublasCusparseNaiveImplementation.cuh"

//////////////////////////////
//////////////////////////////

__global__ void incrementByValue_n(float *arr, int idx, float value) {
    arr[idx] += value;
}


__global__ void vectorScalarMul_n(const float scalar, float *vector, int array_len) {

    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] *= scalar;
        i += gridSize;
    }
}
__global__ void vectorScalarAdd_n(float scalar, float *vector, int array_len) {

    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] += scalar;
        i += gridSize;
    }
}

__global__ void init_vector_n(float* v, int size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        v[i] = value;
        i += gridDim.x * blockDim.x;
    }
}

__global__ void copy_vector_n(float* dest, int* source, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        dest[i] = source[i];
        i += gridDim.x * blockDim.x;
    }
}

//////////////////////////////
//////////////////////////////


void CublasCusparseNaiveImplementation::dotCublas(const float *array1, const float *array2, float *result, int vector_len, float alpha, cublasHandle_t* handle){

    cublasSdot (*handle, vector_len,
                         array1, 1,
                         array2, 1,
                         result);

    *result=alpha*(*result)/vector_len;
}

void CublasCusparseNaiveImplementation::sqCublas(const float *old,float *newVector, float* result, int vector_len,cublasHandle_t* handle){

    float A = -1.0;
    const float* pA = &A;

    cublasSaxpy(*handle, vector_len,
                         pA,
                         old, 1,
                         newVector, 1);
    cudaDeviceSynchronize();

    //if (status!= CUBLAS_STATUS_SUCCESS){
    //    printf("Error cudblast %d\n",static_cast<int>(status));
    //}

    cublasSdot (*handle, vector_len,
                         newVector, 1,
                         newVector, 1,
                         result);

    //if (status!= CUBLAS_STATUS_SUCCESS){
    //    printf("Error cudblast %d\n",static_cast<int>(status));
    //}
}


void CublasCusparseNaiveImplementation::alloc() {
    // Load the input graph and preprocess it;
    pPpr->initialize_graph();
    bsr.blocksInMat = (pPpr->V + bsr.bsrBlockDim - 1) / bsr.bsrBlockDim;//Number of blocks in the matrix, composed by blocksInMat*bsrBlockDim blocks

    cudaMallocManaged(&coo.x_gpu, sizeof(int) * pPpr->E);
    cudaMallocManaged(&csr.rowIndex, sizeof(int) * pPpr->E);
    cudaMallocManaged(&coo.y_gpu, sizeof(int) * pPpr->E);
    cudaMallocManaged(&coo.val_gpu, sizeof(float ) * pPpr->E);


    cudaMallocManaged(&dangling_gpu, sizeof(int) * pPpr->V);
    cudaMallocManaged(&pSquareError_gpu, sizeof(float ));

    cudaMallocManaged(&pr_gpu, sizeof(float)*pPpr->V);
    cudaMallocManaged(&pr_temp, sizeof(float)*pPpr->V+bsr.bsrBlockDim);
    cudaMallocManaged(&pr_old, sizeof(float)*pPpr->V+bsr.bsrBlockDim);

    cudaMallocManaged(&pDanglingFact_gpu, sizeof(float ));
    cudaMallocManaged(&pAlpha_gpu, sizeof(float ));
    cudaMallocManaged(&pTeleportFact_gpu, sizeof(float ));
    cudaMallocManaged(&pV_gpu, sizeof(int ));
    cudaMallocManaged(&pE_gpu, sizeof(int ));
    cudaMallocManaged(&pPersonalization_vertex_gpu, sizeof(int ));
}

//Print error if the block size is too small
void CublasCusparseNaiveImplementation::checkBlockSize(int blockSize,int minBlockSize) {
    if (blockSize < minBlockSize && debug) {
        printf("The selected number of blocks is too small, %d will be used\n",minBlockSize);
    }
}


void CublasCusparseNaiveImplementation::checkCuSparseStatus(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::string error_string = cusparseGetErrorString(status);
        printf("Cusparse error: %s\n", error_string.c_str());
    }
}


void CublasCusparseNaiveImplementation::init() {
    cublasCreate(&cublasHandle);

    cusparseCreate(&cusparseHandle);

    valFloat.resize(pPpr->E);
    std::transform(pPpr->val.begin(), pPpr->val.end(), valFloat.begin(), [](double x) { return (float )x;});

    blocksVertex = 1+pPpr->V/pPpr->T;
    blocksEdge = 1+pPpr->E/pPpr->T;

    int blocks = std::max(blocksVertex,blocksEdge);

    checkBlockSize(pPpr->B,blocks);

    cudaMemset(pr_gpu,0.0, sizeof(float)*pPpr->V);
    cudaMemset(pr_temp,0.0, sizeof(float)*pPpr->V+bsr.bsrBlockDim);

    cudaMemcpy(coo.x_gpu,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice);
    cudaMemcpy(coo.y_gpu,pPpr->y.data(), sizeof(int) * pPpr->E,cudaMemcpyHostToDevice);
    cudaMemcpy(coo.val_gpu,valFloat.data(), sizeof(float ) * pPpr->E,cudaMemcpyHostToDevice);
    vectorScalarMul_n<<<blocksEdge, pPpr->T>>>(pPpr->alpha, coo.val_gpu, pPpr->E);
    cudaDeviceSynchronize();

    cudaMallocManaged((void**)&bsr.rowPtrs, sizeof(int) * (bsr.blocksInMat + 1));
    cooToBsr();
}

void CublasCusparseNaiveImplementation::reset() {
    cudaMemcpy(coo.x_gpu,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice);

    cudaMemcpy(coo.y_gpu,pPpr->y.data(), sizeof(int) * pPpr->E,cudaMemcpyHostToDevice);

    cudaMemcpy(coo.val_gpu,valFloat.data(), sizeof(float ) * pPpr->E,cudaMemcpyHostToDevice);
    vectorScalarMul_n<<<blocksEdge, pPpr->T>>>(pPpr->alpha, coo.val_gpu, pPpr->E);

    cudaMemcpy(dangling_gpu, pPpr->dangling.data(), sizeof(int) * pPpr->V, cudaMemcpyHostToDevice);

    cudaMemset(pSquareError_gpu,INITIAL_SQUARE_ERROR, sizeof(float ));

    cudaMemset(pr_gpu,0.0, sizeof(float)*pPpr->V);
    cudaMemset(pr_temp,0.0, sizeof(float)*bsr.blocksInMat*bsr.bsrBlockDim);
    cudaMemset(pr_old,0.0, sizeof(float)*bsr.blocksInMat*bsr.bsrBlockDim);

    cudaMemset(pDanglingFact_gpu,0, sizeof(float ));
    cudaMemcpy(pAlpha_gpu,&(pPpr->alpha), sizeof(float ),cudaMemcpyHostToDevice);
    float tempTeleportFact = pPpr->alpha/pPpr->V;
    cudaMemcpy(pTeleportFact_gpu,&tempTeleportFact, sizeof(float ),cudaMemcpyHostToDevice);

    cudaMemcpy(pV_gpu,&(pPpr->V), sizeof(int ),cudaMemcpyHostToDevice);
    cudaMemcpy(pE_gpu,&(pPpr->E), sizeof(int ),cudaMemcpyHostToDevice);
    cudaMemcpy(pPersonalization_vertex_gpu,&(pPpr->personalization_vertex), sizeof(int ),cudaMemcpyHostToDevice);

    init_vector_n<<<blocksVertex,pPpr->T>>>(pr_old,pPpr->V,1.0f/float(pPpr->V));

}

void CublasCusparseNaiveImplementation::execute(int iter) {

    if (debug) {
        printf("\nx_gpu initial value\n");
        print_gpu_array(coo.x_gpu, pPpr->E);
        printf("\ny_gpu initial value\n");
        print_gpu_array(coo.y_gpu, pPpr->E);
        printf("\nval_gpu initial value\n");
        print_gpu_array(coo.val_gpu, pPpr->E);
        printf("\npr_old initial value\n");
        print_gpu_array(pr_old, pPpr->V);
    }

    float squareError_cpu = INITIAL_SQUARE_ERROR;

    cudaMemset(pDanglingFact_gpu,0, sizeof(float ));
    cudaMemset(pr_temp,0, sizeof(float )*bsr.blocksInMat*bsr.bsrBlockDim);
    cudaMemset(pSquareError_gpu,0, sizeof(float ));

    for (int i = 0; squareError_cpu > pPpr->convergence_threshold && i < pPpr->max_iterations; i++) {
        if (debug){
            printf("\nConvergence iteration %i pr_old\n",i);
            print_gpu_array(pr_old,pPpr->V);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_old,pPpr->V);}

        cudaMemset(pDanglingFact_gpu,0, sizeof(float ));
        cudaMemset(pr_temp,0, sizeof(float )*bsr.blocksInMat*bsr.bsrBlockDim);
        cudaMemset(pSquareError_gpu,0, sizeof(float ));

        thrust::device_vector<float> danglingFloat(pPpr->V);
        copy_vector_n<<<blocksVertex,pPpr->T>>>(thrust::raw_pointer_cast(danglingFloat.data()),dangling_gpu,pPpr->V);
        dotCublas(thrust::raw_pointer_cast(danglingFloat.data()), pr_old, pDanglingFact_gpu, *pV_gpu, pPpr->alpha,
                  &cublasHandle);
        cudaDeviceSynchronize();

        if (debug)
            printf("\nIteration %i dangling is: %f \n",i,*pDanglingFact_gpu);


        float alpha = 1.0f;
        float beta = 0.0f;
        //size_t bufferSize = pPpr->V * sizeof(float);
        /*
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
        if (bufferSize>0) {//Naive way to create a buffer for coo alg
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

        */

        checkCuSparseStatus(cusparseSbsrmv(
                cusparseHandle, bsr.dir, CUSPARSE_OPERATION_NON_TRANSPOSE, bsr.blocksInMat, bsr.blocksInMat, bsr.nnZBlocks,
                &alpha,
                bsr.bsr_desc,
                bsr.val, bsr.rowPtrs,bsr.colIdxs, bsr.bsrBlockDim,
                pr_old, &beta,pr_temp ));


        cudaDeviceSynchronize();

        cudaDeviceSynchronize();

        if (debug) {
            printf("\npr_temp after coo\n");
            print_gpu_array(pr_temp, pPpr->V);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_temp,pPpr->V);
        }

        vectorScalarAdd_n<<<blocksVertex,pPpr->T>>>(*pDanglingFact_gpu,pr_temp,*pV_gpu);
        cudaDeviceSynchronize();

        if(debug) {
            printf("\npr_temp after adding dangling\n");
            print_gpu_array(pr_temp, pPpr->V);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_temp, pPpr->V);
        }

        //if (noPersonalizedVertex) {
        //    vectorScalarAdd_n<<<blocksVertex,pPpr->T>>>((1.0 - pPpr->alpha)/pPpr->V,pr_temp,*pV_gpu);
        //}else {
            incrementByValue_n<<<1, 1>>>(pr_temp, pPpr->personalization_vertex, 1.0 - pPpr->alpha);
        //}
        cudaMemcpy(pr_gpu,pr_temp ,sizeof(float)*pPpr->V,cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();

        if (debug) {
            printf("\npr_gpu after increment of 1-alpha in position %d\n", pPpr->personalization_vertex);
            print_gpu_array(pr_gpu, 20);
            printf("\nSum is\n");
            printGpu_vector_sum(pr_gpu, pPpr->V);
        }

        //squareError
        //compute_square_error_gpu<<<blocksVertex, pPpr->T>>>(pr_old, pr_gpu,pSquareError_gpu, pPpr->V);
        //pr_temp will be overwritten
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

void CublasCusparseNaiveImplementation::clean() {
    cudaFree(coo.x_gpu);
    cudaFree(coo.y_gpu);
    cudaFree(coo.val_gpu);

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

void CublasCusparseNaiveImplementation::cooToBsr() {
    checkCuSparseStatus(cusparseCreateDnVec(&coo.x_descr,pPpr->V,coo.x_gpu,CUDA_R_32F));
    checkCuSparseStatus(cusparseCreateDnVec(&pr_temp_descr,pPpr->V,pr_temp,CUDA_R_32F));
    checkCuSparseStatus(cusparseCreateDnVec(&pr_old_descr,pPpr->V,pr_old,CUDA_R_32F));


    checkCuSparseStatus(cusparseCreateCoo(
            &coo.descr,
            pPpr->V,
            pPpr->V,
            pPpr->E,
            coo.x_gpu,
            coo.y_gpu,
            coo.val_gpu,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));


    checkCuSparseStatus(cusparseXcoo2csr(
            cusparseHandle,
            coo.x_gpu,pPpr->E,pPpr->V,
            csr.rowIndex,
            CUSPARSE_INDEX_BASE_ZERO)
    );

    /*
    checkCuSparseStatus(cusparseCreateCsr(
            reinterpret_cast<cusparseSpMatDescr_t *>(&mat_desc),

            pPpr->V,pPpr->V,pPpr->E,
            rowIndex,y_gpu,val_gpu,
            CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));*/

    cusparseCreateMatDescr(&csr.mat_desc);
    cusparseSetMatType(csr.mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr.mat_desc, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&bsr.bsr_desc);
    cusparseSetMatType(bsr.bsr_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr.bsr_desc, CUSPARSE_INDEX_BASE_ZERO);

    //Computer number of nonZeroes blocks
    cudaDeviceSynchronize();
    checkCuSparseStatus((cusparseXcsr2bsrNnz(
            cusparseHandle, bsr.dir, pPpr->V, pPpr->V,
            csr.mat_desc, csr.rowIndex, coo.y_gpu, bsr.bsrBlockDim,
            bsr.bsr_desc, bsr.rowPtrs, &bsr.nnZBlocks)));
    cudaMallocManaged((void**)&bsr.colIdxs, sizeof(int) * bsr.nnZBlocks);
    cudaMallocManaged((void**)&bsr.val, sizeof(float) * (bsr.bsrBlockDim * bsr.bsrBlockDim) * bsr.nnZBlocks);
    cudaDeviceSynchronize();
    //Convert from csr to bsr
    checkCuSparseStatus(cusparseScsr2bsr(
            cusparseHandle, bsr.dir, pPpr->V, pPpr->V,
            csr.mat_desc, coo.val_gpu, csr.rowIndex, coo.y_gpu, bsr.bsrBlockDim,
            bsr.bsr_desc, bsr.val, bsr.rowPtrs, bsr.colIdxs));
    cudaDeviceSynchronize();
    /*
    for(auto i = 0; i < min(20,(bsr.blocksInMat + 1)); i++) {
        printf("bsrRowPtrC[%2d] = %d\n", i, bsr.rowPtrs[i]);
    }
    printf("\n");

    for(auto i = 0; i < min(bsr.nnZBlocks,20); i++) {
        printf("bsrColIndC[%2d] = %d\n", i, bsr.colIdxs[i]);
    }
    printf("\n");

    for(auto i = 0; i < min(20,(bsr.bsrBlockDim * bsr.bsrBlockDim) * bsr.nnZBlocks); i++) {
        printf("bsrVal[%2d] = %f\n", i, bsr.val[i]);
    }*/
}





