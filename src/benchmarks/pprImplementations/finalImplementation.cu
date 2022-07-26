#include "finalImplementation.cuh"
#include "../pprFunctions/pprFunctions.cuh"
#include "../pprFunctions/coospmv.cuh"
#include <algorithm>

void FinalImplementation::initDanglingPprTop19(){

    float val = 10.0;

    // California top19
    // int top19[19] = {16,40,65, 210, 996, 1078, 1082, 1487, 1488, 1616, 1805, 1861, 1862, 2077, 2407, 4390, 4822, 6426, 9663};

    // Wikipedia top19
    // 0000007030 0000009008 0000024716 0000027566 0000028020 0000028196 0000195101 0000500469 0000577659 0000689491 0000932394 0001518892 0001702309 0001835017 0002144742 0002257865 0002532493 0002979297 0002984189
    int top19[19] = {7030, 9008, 24716, 27566, 28020, 28196, 195101, 500469, 577659, 689491, 932394, 1518892, 1702309, 1835017, 2144742, 2257865, 2532493, 2979297, 2984189};

    for(int i=0; i<19; i++){
        danglingPprTop19.insert(std::pair<int, double>(top19[i], val));
        val = val-0.1;
    }

}

void FinalImplementation::initDanglingPpr(){
    float val;
    for(int i=0; i<pPpr->V; i++){
        if(danglingPprTop19.count(i)){
            val = danglingPprTop19.find(i)->second;
            danglingPpr.push_back(val);
        }else danglingPpr.push_back(0.0);
    }

}

void FinalImplementation::initDanglingIndexes() {
    for(int i=0; i<pPpr->V; i++){
        if(pPpr->dangling[i] == 1) pDanglingIndexes.push_back(i);
    }
}

void FinalImplementation::alloc(){
    // Load the input graph and preprocess it;

    pPpr->initialize_graph();
    initDanglingIndexes();
    initDanglingPprTop19();
    initDanglingPpr();
    danglingSize = pDanglingIndexes.size();

    cudaMalloc(&x_gpu, sizeof(int) * pPpr->E);
    cudaMalloc(&y_gpu, sizeof(int) * pPpr->E);
    cudaMalloc(&val_gpu, sizeof(float ) * pPpr->E);


    cudaMalloc(&dangling_gpu, sizeof(int) *pPpr->V);
    cudaMalloc(&pSquareError_gpu, sizeof(float));

    cudaMalloc(&pr_gpu, sizeof(float)*pPpr->V);
    cudaMalloc(&pr_old, sizeof(float)*pPpr->V);
    cudaMalloc(&pr_gpu_double, sizeof(double)*pPpr->V);

    cudaMallocManaged(&pDanglingFact_gpu,sizeof(float)) ;
    cudaMalloc(&pDanglingIndexes_gpu, sizeof( int)* danglingSize);

    num_units  = pPpr->E / WARP_SIZE;
    num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    num_blocks = (num_warps + (WARPS_PER_BLOCK - 1)) / WARPS_PER_BLOCK;
    num_iters  = (num_units + (num_warps - 1)) / num_warps;
    interval_size = WARP_SIZE * num_iters;
    tail = num_units * WARP_SIZE ;// do the last few nonzeros separately (fewer than WARP_SIZE elements)
    active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

    cudaMalloc(&temp_rows, sizeof(int) * active_warps);
    cudaMalloc(&temp_vals, sizeof(float) * active_warps);

    reducedV = pPpr->V; // 1.500.000 worked well

    BLOCKS_V = (pPpr->V + THREADS - 1)/ THREADS;
    BLOCKS_E = (pPpr->E + THREADS - 1)/ THREADS;
    BLOCKS_ERROR = (reducedV + THREADS - 1)/ THREADS;
    BLOCKS_D = (danglingSize + THREADS - 1)/ THREADS;
    ppVertexConst = (float)(1.0 - pPpr->alpha);
    dampingFract = (float) (pPpr->alpha / pPpr->V);
}

void FinalImplementation::init() {

    valFloat.resize(pPpr->E);
    std::transform(pPpr->val.begin(), pPpr->val.end(), valFloat.begin(), [](double x) { return (float )x;});

    prFloat.resize(pPpr->V);
    std::transform(pPpr->pr.begin(), pPpr->pr.end(), prFloat.begin(), [](double x) { return (float) x; });

    cudaMemcpy(x_gpu,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu,pPpr->y.data(), sizeof(int) * pPpr->E,cudaMemcpyHostToDevice);

    cudaMemcpy(val_gpu, valFloat.data(), sizeof(float) * pPpr->E, cudaMemcpyHostToDevice);
    vectorScalarMul_math<float><<<BLOCKS_E, THREADS>>>(pPpr->alpha, val_gpu, pPpr->E);

    cudaMemcpy(dangling_gpu, pPpr->dangling.data(), sizeof(int) * pPpr->V, cudaMemcpyHostToDevice);

    cudaMemcpy(pDanglingIndexes_gpu, pDanglingIndexes.data(), sizeof(int) * danglingSize, cudaMemcpyHostToDevice);

}

void FinalImplementation::reset() {

    if (old_ppr_index != -1) danglingPpr[old_ppr_index] = 0.0;

    if(isDangling(pPpr->personalization_vertex)) {
        danglingPpr[pPpr->personalization_vertex] = 20.0;
        old_ppr_index = pPpr->personalization_vertex;
    }

    init_vector<float><<<BLOCKS_V, THREADS>>>(pr_gpu, pPpr->V, 0.0);
    cudaMemcpy(pr_old, prFloat.data(), sizeof(float) * pPpr->V, cudaMemcpyHostToDevice);

}

bool FinalImplementation::isDangling(int vertex){
    return (pPpr->dangling[vertex] == 1);
}

void FinalImplementation::execute(int iter) {

    cudaStream_t coospmv_stream;
    cudaStream_t dangling_stream;
    cudaStream_t euclidean_stream;
    cudaStream_t init_copy_stream;
    cudaStreamCreate(& coospmv_stream);
    cudaStreamCreate(&dangling_stream);
    cudaStreamCreate(&euclidean_stream);
    cudaStreamCreate(&init_copy_stream);

    if(isDangling(pPpr->personalization_vertex)){
        std::copy(danglingPpr.begin(), danglingPpr.end(), pPpr->pr.begin());
        return;
    }
    const int THREADS = 256;
    squareError_cpu = INITIAL_SQUARE_ERROR;
    float heuristic_threshold = 0.0000006;

    for (int i = 0; squareError_cpu > heuristic_threshold && i < pPpr->max_iterations; i++) {

        if (i == 0) init_vector<double><<<BLOCKS_V, THREADS>>>(pr_gpu_double, pPpr->V, 0.0);

        cudaMemsetAsync(pDanglingFact_gpu,0.0, sizeof(float), dangling_stream);
        cudaMemsetAsync(pSquareError_gpu,0.0, sizeof(float), euclidean_stream);
        init_vector<float><<<BLOCKS_V, THREADS, 0, init_copy_stream>>>(pr_gpu, pPpr->V, 0);

        dangling_kernel< int, float><<<BLOCKS_D, THREADS, THREADS * sizeof(float), dangling_stream>>>(pDanglingIndexes_gpu, pr_old, pDanglingFact_gpu, dampingFract,danglingSize);
        __spmv_coo_flat<int, float, THREADS>(x_gpu, y_gpu, val_gpu, pr_old, pr_gpu, pPpr->E, num_blocks, interval_size, tail,
                                             active_warps, temp_rows, temp_vals);

        cudaDeviceSynchronize();

        vectorScalarAddAndIncrement_math<float><<<BLOCKS_V, THREADS>>>(*pDanglingFact_gpu, pr_gpu, pPpr->V, pPpr->personalization_vertex, ppVertexConst);
        cudaDeviceSynchronize();
        if((i > 3 && i < 12) || (i > 12 && i % 3 == 0)){
            euclidean_kernel_math < float ><<<BLOCKS_V, THREADS, THREADS * sizeof(float), euclidean_stream>>>(pr_old, pr_gpu, pSquareError_gpu, pPpr->V);
            cudaMemcpyAsync(&squareError_cpu, pSquareError_gpu, sizeof(float), cudaMemcpyDeviceToHost);
            if(squareError_cpu < heuristic_threshold) squareError_cpu = 0.0;
        }

        copy_vector<float><<<BLOCKS_V, THREADS, 0, init_copy_stream>>>(pr_old,pr_gpu,  pPpr->V);

    }
    cudaDeviceSynchronize();
    cast_vector<double, float><<<BLOCKS_V, THREADS>>>(pr_gpu_double,pr_gpu, pPpr->V);

    cudaStreamDestroy(coospmv_stream);
    cudaStreamDestroy(dangling_stream);
    cudaStreamDestroy(euclidean_stream);
    cudaStreamDestroy(init_copy_stream);

    //A pointer to the address in base class is used since the validation is done by the base class
    cudaMemcpy(pPpr->pr.data(),pr_gpu_double, sizeof (double )*pPpr->V,cudaMemcpyDeviceToHost);

}

void FinalImplementation::clean() {
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(val_gpu);

    cudaFree(dangling_gpu);
    cudaFree(pSquareError_gpu);

    cudaFree(pr_gpu);
    cudaFree(pr_gpu_double);
    cudaFree(pr_old);
    cudaFree(pDanglingFact_gpu);

    cudaFree(pDanglingIndexes_gpu);

    cudaFree(temp_rows);
    cudaFree(temp_vals);
}