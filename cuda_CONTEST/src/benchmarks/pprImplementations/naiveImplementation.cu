#include "naiveImplementation.cuh"
#include "../personalized_pagerank.cuh"
#include "../pprFunctions/pprFunctions.cuh"

void NaiveImplementation::alloc() {
    // Load the input graph and preprocess it;
    pPpr->initialize_graph();

    cudaMallocManaged(&x_gpu, sizeof(int) * pPpr->E);
    cudaMallocManaged(&y_gpu, sizeof(int) * pPpr->E);
    cudaMallocManaged(&val_gpu, sizeof(double ) * pPpr->E);


    cudaMallocManaged(&dangling_gpu, sizeof(int) * pPpr->V);
    cudaMallocManaged(&pSquareError_gpu, sizeof(double ));

    cudaMallocManaged(&pr_gpu, sizeof(double)*pPpr->V);
    cudaMallocManaged(&pr_temp, sizeof(double)*pPpr->V);
    cudaMallocManaged(&pr_old, sizeof(double)*pPpr->V);

    cudaMallocManaged(&pDanglingFact_gpu, sizeof(double ));
    cudaMallocManaged(&pAlpha_gpu, sizeof(double ));
    cudaMallocManaged(&pTeleportFact_gpu, sizeof(double ));
    cudaMallocManaged(&pV_gpu, sizeof(int ));
    cudaMallocManaged(&pE_gpu, sizeof(int ));
    cudaMallocManaged(&pPersonalization_vertex_gpu, sizeof(int ));

    BLOCKS_V = (pPpr->V + THREADS - 1)/ THREADS;
    BLOCKS_E = (pPpr->E + THREADS - 1)/ THREADS;
}

void NaiveImplementation::init() {
}

void NaiveImplementation::reset() {

    cudaMemcpy(x_gpu,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice);

    cudaMemcpy(y_gpu,pPpr->y.data(), sizeof(int) * pPpr->E,cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu,pPpr->val.data(), sizeof(double ) * pPpr->E,cudaMemcpyHostToDevice);
    vectorScalarMul<double><<<BLOCKS_E, THREADS>>>(pPpr->alpha, val_gpu, pPpr->E);

    cudaMemcpy(dangling_gpu, pPpr->dangling.data(), sizeof(int) * pPpr->V, cudaMemcpyHostToDevice);
    cudaMemset(pr_gpu,0.0, sizeof(double)*pPpr->V);
    cudaMemset(pr_temp,0.0, sizeof(double)*pPpr->V);
    cudaMemcpy(pr_old,pPpr->pr.data() ,sizeof(double)*pPpr->V,cudaMemcpyHostToDevice);

    cudaMemset(pDanglingFact_gpu,0.0, sizeof(double ));
    cudaMemcpy(pAlpha_gpu,&(pPpr->alpha), sizeof(double ),cudaMemcpyHostToDevice);
    double tempTeleportFact = pPpr->alpha/pPpr->V;
    cudaMemcpy(pTeleportFact_gpu,&tempTeleportFact, sizeof(double ),cudaMemcpyHostToDevice);

    cudaMemcpy(pV_gpu,&(pPpr->V), sizeof(int ),cudaMemcpyHostToDevice);
    cudaMemcpy(pE_gpu,&(pPpr->E), sizeof(int ),cudaMemcpyHostToDevice);
    cudaMemcpy(pPersonalization_vertex_gpu,&(pPpr->personalization_vertex), sizeof(int ),cudaMemcpyHostToDevice);
}

void NaiveImplementation::execute(int iter) {

    double squareError_cpu = INITIAL_SQUARE_ERROR;

    cudaMemset(pDanglingFact_gpu,0.0, sizeof(double ));
    cudaMemset(pr_temp,0.0, sizeof(double )*pPpr->V);
    cudaMemset(pSquareError_gpu,0.0, sizeof(double ));

    for (int i = 0; squareError_cpu > pPpr->convergence_threshold && i < pPpr->max_iterations; i++) {

        cudaMemset(pDanglingFact_gpu,0.0, sizeof(double));
        init_vector<double><<<BLOCKS_V, THREADS>>>(pr_temp,pPpr->V, 0.0);
        cudaMemset(pSquareError_gpu,0.0, sizeof(double ));
        cudaDeviceSynchronize();

        compute_dangling_factor_gpu<int, double><<<BLOCKS_V, THREADS>>>(dangling_gpu, pr_old, pDanglingFact_gpu, *pV_gpu);
        cudaDeviceSynchronize();

        cooSPMV<int, double><<<BLOCKS_E,THREADS>>>(x_gpu,y_gpu,val_gpu,pPpr->E,pr_old,pr_temp);
        cudaDeviceSynchronize();

        vectorScalarAdd<double><<<BLOCKS_V,THREADS>>>(pDanglingFact_gpu[0] * (pPpr->alpha / pPpr->V),pr_temp,*pV_gpu);
        cudaDeviceSynchronize();

        copy_vector<double><<<BLOCKS_V, THREADS>>>(pr_gpu,pr_temp, pPpr->V);
        cudaDeviceSynchronize();

        incremetBy1<double><<<1, 1>>>(pr_gpu, pPpr->personalization_vertex,1.0-pPpr->alpha);
        cudaDeviceSynchronize();

        compute_square_error_gpu<double><<<BLOCKS_V, THREADS>>>(pr_old, pr_gpu, pSquareError_gpu, pPpr->V);
        cudaDeviceSynchronize();

        cudaMemcpy(&squareError_cpu,pSquareError_gpu,sizeof (double ),cudaMemcpyDeviceToHost);

        squareError_cpu= std::sqrt(squareError_cpu);

        copy_vector<double><<<BLOCKS_V, THREADS>>>(pr_old,pr_gpu, pPpr->V);
        cudaDeviceSynchronize();

    }

    cudaDeviceSynchronize();
    //A pointer to the address in base class is used since the validation is done by the base class
    cudaMemcpy(pPpr->pr.data(),pr_gpu,sizeof(double)*pPpr->V,cudaMemcpyDeviceToHost);
}

void NaiveImplementation::clean() {
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(val_gpu);
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
}
