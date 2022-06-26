#include "improvedImplementation.cuh"
#include "../personalized_pagerank.cuh"
#include "../pprFunctions/pprFunctions.cuh"

//convert COO in CSR
void ImprovedImplementation::initCSR() {
    int ptr = 0, previousX;

    if (pPpr->E == 0)
        return;

    previousX = 0;
    xPtr.push_back(0);

    for (int i = 0; i < pPpr->E; i++) {
        while (pPpr->x[i] != previousX) {
            xPtr.push_back(ptr);
            previousX++;
        }
        ptr++;
    }

    for (int i = 0; i < pPpr->V - pPpr->x[pPpr->E - 1]; i++) {
        xPtr.push_back(ptr);
    }
}

void ImprovedImplementation::alloc(){
    // Load the input graph and preprocess it;
    pPpr->initialize_graph();
    initCSR();

    cudaMallocManaged(&x_gpu, sizeof(int) * pPpr->E);
    cudaMallocManaged(&xPtr_gpu, sizeof(int) * (pPpr->V+1));
    cudaMallocManaged(&y_gpu, sizeof(int) * pPpr->E);
    cudaMallocManaged(&val_gpu, sizeof(double ) * pPpr->E);


    cudaMallocManaged(&dangling_gpu, sizeof(int) *pPpr->V);
    cudaMallocManaged(&pSquareError_gpu, sizeof(double));

    cudaMallocManaged(&pr_gpu, sizeof(double)*pPpr->V);
    cudaMallocManaged(&pr_temp, sizeof(double)*pPpr->V);
    cudaMallocManaged(&pr_old, sizeof(double)*pPpr->V);
    cudaMallocManaged(&prMinus2, sizeof(double)*pPpr->V);
    cudaMallocManaged(&g, sizeof(double)*pPpr->V);
    cudaMallocManaged(&h, sizeof(double)*pPpr->V);

    cudaMallocManaged(&pDanglingFact_gpu,sizeof(double)) ;
    cudaMallocManaged(&count1, sizeof(unsigned int));
    cudaMallocManaged(&count2, sizeof(unsigned int));

    BLOCKS_V = (pPpr->V + THREADS - 1)/ THREADS;
    BLOCKS_E = (pPpr->E + THREADS - 1)/ THREADS;
}

void ImprovedImplementation::init() {

}

void ImprovedImplementation::reset() {

    cudaMemcpy(x_gpu,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice);

    cudaMemcpy(xPtr_gpu,xPtr.data(), sizeof(int) * (pPpr->V + 1), cudaMemcpyHostToDevice);

    cudaMemcpy(y_gpu,pPpr->y.data(), sizeof(int) * pPpr->E,cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu,pPpr->val.data(), sizeof(double ) * pPpr->E,cudaMemcpyHostToDevice);
    vectorScalarMul<double><<<BLOCKS_E, THREADS>>>(pPpr->alpha, val_gpu, pPpr->E);

    cudaMemcpy(dangling_gpu, pPpr->dangling.data(), sizeof(int) * pPpr->V, cudaMemcpyHostToDevice);

    cudaMemset(pr_gpu,0.0, sizeof(double)*pPpr->V);
    cudaMemset(pr_temp,0.0, sizeof(double)*pPpr->V);
    cudaMemcpy(pr_old,pPpr->pr.data() ,sizeof(double)*pPpr->V,cudaMemcpyHostToDevice);



}

void ImprovedImplementation::execute(int iter) {
    double squareError_cpu = INITIAL_SQUARE_ERROR;
    int aitken = 0;

    double dampingFract = (double) pPpr->alpha / pPpr->V;

    for (int i = 0; squareError_cpu > pPpr->convergence_threshold && i < pPpr->max_iterations; i++) {

        init_vector<double><<<BLOCKS_V, THREADS>>>(pr_gpu, pPpr->V, 0);
        init_vector<double><<<BLOCKS_V, THREADS>>>(pDanglingFact_gpu, 1, 0);
        init_vector<double><<<BLOCKS_V, THREADS>>>(pSquareError_gpu, 1, 0);

        cudaDeviceSynchronize();

        if (!aitken) {

            dot_product_kernel<int, double><<<BLOCKS_V, THREADS, THREADS * sizeof(double)>>>(dangling_gpu, pr_old, pDanglingFact_gpu, dampingFract, pPpr->V);
            cooSPMV<int, double><<<BLOCKS_E, THREADS>>>(x_gpu, y_gpu, val_gpu, pPpr->E, pr_old, pr_gpu); // needs improvement!!
            cudaDeviceSynchronize();

            vectorScalarAddAndIncrement<double><<<BLOCKS_V, THREADS>>>(*pDanglingFact_gpu, pr_gpu,
                                                                       pPpr->V, pPpr->personalization_vertex,
                                                                       1.0 - pPpr->alpha);
            cudaDeviceSynchronize();

            compute_square_error_gpu<double><<<BLOCKS_V, THREADS, THREADS * sizeof(double)>>>(pr_old, pr_gpu, pSquareError_gpu, pPpr->V);

            cudaMemcpy(&squareError_cpu, pSquareError_gpu, sizeof(double), cudaMemcpyDeviceToHost);
            squareError_cpu = std::sqrt(squareError_cpu);


            if (i == 30) cudaMemcpy(prMinus2, pr_old, sizeof(double) * pPpr->V, cudaMemcpyDeviceToDevice);


            if (i==30)  aitken = 1;

        }

        if(aitken){
            compute_aikten_x_math<<<BLOCKS_V, THREADS>>>(pr_gpu, pr_old, prMinus2, pPpr->V);
            aitken = 0;
            cudaDeviceSynchronize();
        }

        cudaMemcpy(pr_old, pr_gpu, sizeof(double) * pPpr->V, cudaMemcpyDeviceToDevice);

    }

    cudaDeviceSynchronize();

    //A pointer to the address in base class is used since the validation is done by the base class
    cudaMemcpy(pPpr->pr.data(),pr_gpu,sizeof (double )*pPpr->V,cudaMemcpyDeviceToHost);
}

void ImprovedImplementation::clean() {
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(val_gpu);
    cudaFree(dangling_gpu);
    cudaFree(pSquareError_gpu);
    cudaFree(pr_gpu);
    cudaFree(pr_temp);
    cudaFree(pr_old);
    cudaFree(pDanglingFact_gpu);
    cudaFree(count1);
    cudaFree(count2);
}
