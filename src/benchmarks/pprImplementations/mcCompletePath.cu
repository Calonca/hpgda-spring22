#include <queue>
#include <random>
#include "../personalized_pagerank.cuh"
#include "mcCompletePath.cuh"

#define curandErrCheck(ans) { gpuAssertrand((ans), __FILE__, __LINE__); }
inline void gpuAssertrand(curandStatus code, const char *file, int line, bool abort=true)
{
    if (code != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr,"curand error: %s %d\n", file, line);
        if (abort) exit(code);
    }
}

//////////////////////////////
//////////////////////////////

__global__ void init_pr(float* pr, const int* idxs,int idxsSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < idxsSize) {
        pr[idxs[i]] = 1.0f/float(i+2);
    }
}


/* This kernel initializes state per thread for each of x, y, and z */
#define VECTOR_SIZE 32
__global__ void setup_kernel(unsigned int * sobolDirectionVectors,
                             unsigned int *sobolScrambleConstants,
                             curandStateScrambledSobol32 *state)
{

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(sobolDirectionVectors +VECTOR_SIZE*id,
                sobolScrambleConstants[id],
                1234,
                &state[id]);

}

__global__ void setup_curand(curandState* states, unsigned long long seed) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void setup_curand(curandStatePhilox4_32_10_t* states, unsigned long long seed) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void setup_curand(curandStateMRG32k3a* states, unsigned long long seed) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}

__device__ unsigned int generateInt(curandStateScrambledSobol32 *state)
{
    /* Copy state to local memory for efficiency */
    curandStateScrambledSobol32 localState = *state;
    /* Generate pseudo-random unsigned ints */
    unsigned int x = curand(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}

__device__ unsigned int generateInt(curandState *state)
{
    /* Copy state to local memory for efficiency */
    curandState localState = *state;
    /* Generate pseudo-random unsigned ints */
    unsigned int x = curand(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}

__device__ unsigned int generateInt(curandStatePhilox4_32_10_t *state)
{
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = *state;
    /* Generate pseudo-random unsigned ints */
    unsigned int x = curand(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}

__device__ unsigned int generateInt(curandStateMRG32k3a *state)
{
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = *state;
    /* Generate pseudo-random unsigned ints */
    unsigned int x = curand(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}


__device__ float generateFloat(curandState *state)
{
    /* Copy state to local memory for efficiency */
    curandState localState = *state;
    /* Generate pseudo-random uniforms */
    float x = curand_uniform(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}

__device__ float generateFloat(curandStatePhilox4_32_10_t *state)
{
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = *state;
    /* Generate pseudo-random uniforms */
    float x = curand_uniform(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}

__device__ float generateFloat(curandStateMRG32k3a *state)
{
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = *state;
    /* Generate pseudo-random uniforms */
    float x = curand_uniform(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}
__device__ float generateFloat(curandStateScrambledSobol32 *state)
{
    /* Copy state to local memory for efficiency */
    curandStateScrambledSobol32 localState = *state;
    /* Generate pseudo-random unsigned ints */
    float x = curand_uniform(&localState);
    /* Copy state back to global memory */
    *state = localState;
    return x;
}



__global__ void TestCurand(
        const int maxWalkLen,
        const int walkers,
        curandState* states
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("i is %d \n", i);
    if (i < walkers) {
        int cumulativeWalks = 0;
        while (cumulativeWalks < maxWalkLen) {
            float rand0 = generateFloat(&states[i]);
            int rand10 = (int) (rand0 * 10);
            printf("Rand nums %d are: %d for walker %d \n",cumulativeWalks, rand10, i);
            cumulativeWalks++;
        }
        //printf("Adding to %d for walker %d, walkLen is %d\n", vIdx, i, walkLen);
        //pr[vIdx]+=1.0f;

    }
}

/// Monte Carlo complete path algorithm
/// Adding 1 to each node visited starting from the seed s
/// A walker has probability 1-alpha of stopping at each node and alpha of continuing.
/// if the walker continues it chooses a random node from the list of neighbors.
/// \param pr the page rank vector
/// \param x indices of the destinations of each edge in a csc matrix
/// \param xPtr pointer to destination indexes of each vertex in a csc matrix
/// \param s personalidex vertex
/// \param stopThreshold if the pseudorandom probaility is less than this the the walker will stop at the current node
/// \param spawnWalkerThreshold if the number of steps is lower than this number, a new walker is spawned at the seed,
/// in this way if we don't have idle threads
/// \param maxWalkLen the maximum number of steps performed before the algorithm stops
/// \param walkers the number of concurrent walkers or threads that starting from the seed s explore the graph
/// \param states states for the pseudo random number generator
__global__ void MCompletePath(
        float* pr,
        const int* x,
        const int* xPtr,
        const int s,
        float stopThreshold,
        const int spawnWalkerThreshold,
        const int maxWalkLen,
        const int walkers,
        const int neighStartPos,
        const int* neighSizee,
        curandStatePhilox4_32_10_t * states
        ) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = states[i];
    if (i < walkers) {

        //Starting from neighbors of s
        unsigned int rIdx0 = curand(&localState) % neighSizee[s];
        int vIdx =  x[neighStartPos + rIdx0];
        int cumulativeWalks = 1;

        while (cumulativeWalks < maxWalkLen) {
            atomicAdd(&pr[vIdx], 1.0f);//Add 1 to each visited node
            float rand0 = curand_uniform(&localState);
            if (rand0 < stopThreshold) {//Terminate with probability 1-alpha
                if (cumulativeWalks<spawnWalkerThreshold)
                    vIdx=s;
                else
                    break;
            } else {
                int neighStartIdx = xPtr[vIdx];
                if (neighSizee[vIdx] == 0) {
                    if (cumulativeWalks<spawnWalkerThreshold)
                        vIdx=s;
                    else
                        break;
                }else {
                    //Select a random node from the list of neighbors
                    unsigned int rIdx = curand(&localState) % neighSizee[vIdx];
                    vIdx = x[neighStartIdx + rIdx];
                }
            }
            cumulativeWalks++;
        }
    }
    //Copy state back to global memory
    states[i] = localState;
}

//////////////////////////////
//////////////////////////////

void MCCompletePath::alloc() {
    // Load the input graph and preprocess it;
    pPpr->initialize_graph();

    errCheck(cudaMalloc(&coo.x, sizeof(int) * pPpr->E));

    errCheck(cudaMalloc(&csc.x, sizeof(int) * pPpr->E));
    errCheck(cudaMalloc(&csc.xPtr, sizeof(int) * (pPpr->V+1)));
    errCheck(cudaMalloc(&csc.neightSize, sizeof(int) * (pPpr->V)));

    errCheck(cudaMalloc(&pr_gpu, sizeof(float) * pPpr->V));
    errCheck(cudaMalloc(&top19Gpu, sizeof(int)*19));

    walkers = pPpr->B * pPpr->T;

    errCheck(cudaMalloc (&statesPhilox, walkers * sizeof(curandStatePhilox4_32_10_t)));

}

int californiaTop19[19] = {1487, 4390, 65, 6426, 4822, 2077, 9663, 1488,1616, 2407, 16, 1805, 996, 40, 210, 1862, 1861, 1082, 1078};
float californiaVal[19] = {0.005296, 0.005172, 0.004057, 0.003928, 0.003852, 0.003691, 0.003568, 0.00337, 0.003098, 0.003089, 0.003035, 0.002692, 0.00267, 0.002548, 0.002538, 0.002492, 0.002492, 0.002344, 0.002339};
int wikiTop19[19] = {1702309, 500469, 1518892, 24716, 689491, 195101, 932394, 2984189, 1835017, 577659, 28196, 7030,2257865, 2532493, 28020, 2979297, 9008,27566 , 2144742};
const int CALIFORNIA = 9664;
const int WIKI = 3566907;
void MCCompletePath::initPr(bool initGPU, bool initCPU) {

    int* top19;// = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    if (pPpr->V==CALIFORNIA || pPpr->V==WIKI) {
        if (pPpr->V==WIKI){
            top19 = wikiTop19;
        }else   {
            top19 = californiaTop19;
        }
        for (int i=0;i<19;i++) {
            if (initCPU)
                pPpr->pr[top19[i]] = 1.0f/float(i+2);
        }
        if (initGPU) {
            errCheck(cudaMemcpy(top19Gpu,top19, sizeof(int)*19, cudaMemcpyHostToDevice));
            //init_pr<<<1, 19>>>(initialPr_gpu, top19Gpu, 19);
            init_pr<<<1, 19>>>(pr_gpu, top19Gpu, 19);
            errCheck(cudaDeviceSynchronize());
            //initialPr_gpu[top19[i]] = 1.0f;
        }
    }
}

void MCCompletePath::init() {

    stopThreshold = 1-pPpr->alpha;
    maxWalkLen = std::min(pPpr->max_iterations,50);
    spawnWalkerThreshold = maxWalkLen - 30;

    errCheck(cudaMemcpy(coo.x,pPpr->x.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice));

    prFloat.resize(pPpr->V);

    sortAndConvert();
    errCheck(cudaMemcpy(csc.x,csc.x_cpu.data(), sizeof(int) * pPpr->E, cudaMemcpyHostToDevice));
    errCheck(cudaMemcpy(csc.xPtr,csc.xPtr_cpu.data(), sizeof(int) * (pPpr->V+1), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpy(csc.neightSize,csc.neightSize_cpu.data(), sizeof(int) * (pPpr->V), cudaMemcpyHostToDevice));

    srand(time(0));
    int seed = rand();
    //setup_curand<<<pPpr->B,pPpr->T>>>(states,seed);
    //errCheck(cudaPeekAtLastError());
    //errCheck(cudaDeviceSynchronize());
    setup_curand<<<pPpr->B,pPpr->T>>>(statesPhilox,seed);
    errCheck(cudaPeekAtLastError());
    errCheck(cudaDeviceSynchronize());
    //setup_curand<<<pPpr->B,pPpr->T>>>(statesMRG,seed);
    //errCheck(cudaPeekAtLastError());
    //errCheck(cudaDeviceSynchronize());

}

void MCCompletePath::reset() {
    ///Alt reset
    errCheck(cudaMemset(pr_gpu, 0, sizeof(float)*pPpr->V));
    initPr(true, true);
    errCheck(cudaDeviceSynchronize());
}



void MCCompletePath::execute(int iter) {

    bool testFailures = false;
    if (pPpr->V==WIKI && testFailures) {
        if (iter==0) {
            //pPpr->personalization_vertex = 3518145;//A dangling
            // pPpr->personalization_vertex = 3562862;//Near dangling
            pPpr->personalization_vertex = 2091850;
        }
    }

    int s = pPpr->personalization_vertex;
    //dangling 3518145
    /*
    std::vector<int> pvToTest;
    pvToTest.push_back(3518145);
    if (iter<pvToTest.size())
        s=pvToTest[iter];*/

    if (pPpr->dangling[s]){
        pPpr->pr[s] = 1.0f;
        if (debug) {
            printf("The seed vertex %d is dangling\n", s);
        }
        return;
    }
    /*
    std::cout << "\nPr gpu: " << std::endl;
    print_gpu_array(pr_gpu,pPpr->V);

    std::cout << "\nx csc cpu " << std::endl;
    print_array(csc.x_cpu.data(),pPpr->E);
    std::cout << "\nx csc " << std::endl;
    print_gpu_array(csc.x,pPpr->E);

    std::cout << "\ncsc cpu pointers" << std::endl;
    print_array(csc.xPtr_cpu.data(),pPpr->E);
    std::cout << "\ncsc pointers " << std::endl;
    print_gpu_array(csc.xPtr,pPpr->E);*/
    //MCCompletePathAlgoCPU(pPpr->personalization_vertex, walkers);
    //return;
    int neighStartIdx = csc.xPtr_cpu[s];

    MCompletePath<<<pPpr->B, pPpr->T>>>
            (pr_gpu, csc.x, csc.xPtr, s, stopThreshold, spawnWalkerThreshold, maxWalkLen, walkers,
             neighStartIdx, csc.neightSize,
             statesPhilox);

    errCheck(cudaPeekAtLastError());
    errCheck(cudaDeviceSynchronize());

    errCheck(cudaMemcpy(prFloat.data(),pr_gpu, sizeof(float )*pPpr->V,cudaMemcpyDeviceToHost));
    std::transform(prFloat.begin(), prFloat.end(), pPpr->pr.begin(), [](float x) { return (double )x;});
    pPpr->pr[s] += walkers;
}


void MCCompletePath::clean() {
    cudaFree(coo.x);
    cudaFree(csc.x);
    cudaFree(csc.xPtr);
    cudaFree(csc.neightSize);
    cudaFree(pr_gpu);
    cudaFree(states);
    cudaFree(statesPhilox);
    cudaFree(statesMRG);
}



void MCCompletePath::getKClosest(int k, std::vector<int> &closest, int pVertex) {
    //For each vertex add it to coo.adjList

    std::vector<bool> visited(pPpr->V, false);
    std::queue<int> queue;

    queue.push(pVertex);

    int foundK = 0;

    if (pPpr->dangling[pVertex] == 1) {
        //Print is dangling
        std::cout << "Is dangling vertex " << std::endl;
        return;
    }

    while(foundK<k && !queue.empty()) {
        auto elms = coo.adjList[queue.front()];
        queue.pop();
        for (auto v : elms) {
            printf("Checking y:%d \n", v);
            if (!visited[v]) {
                printf("Adding y:%d \n", v);
                closest.push_back(v);
                foundK++;
                visited[v] = true;
                queue.push(v);
            }
        }

    }
    //print pVertex
    std::cout << "Personalized vertex "<<pVertex<< std::endl;
    //print closest
    for (auto el : closest) {

        printf("Closest to pv y:%d \n", el);
    }

}

void MCCompletePath::MCCompletePathAlgoCPU(int s, int nWalkers) {

    initPr(false, true);
    std::random_device rd;
    std::mt19937 mt(rd());
    //number between 0 and 1
    std::uniform_real_distribution<float> randProb(0, 1);
    std::uniform_int_distribution<int> randInt(0, pPpr->V - 1);


    int walkLen = 0;
    for (int run = 0; run < nWalkers; run++) {
        int vIdx = s;
        while (walkLen < maxWalkLen){
            pPpr->pr[vIdx]+=1.0f;
            bool terminate = randProb(mt) < 1.0f - pPpr->alpha;//Terminate with probability 1-alpha
            if (terminate) {
                break;
            } else {
                int outVStartIdx = csc.xPtr_cpu[vIdx];
                int outVEndIdx = csc.xPtr_cpu[vIdx + 1];
                int outEdgesSize = outVEndIdx - outVStartIdx;
                if (outEdgesSize == 0) {
                    break;}

                //select a random connected vertex
                int rIdx = randInt(mt) % outEdgesSize;
                vIdx=csc.x_cpu[outVStartIdx + rIdx];
            }
            walkLen++;
        }
        walkLen = 0;
    }
    if (debug) {
        std::cout << "MC cpu pr: " << std::endl;
        //std::sort(pr_test.begin(), pr_test.end());
        print_array(pPpr->pr.data(), pPpr->V);
    }
}

void MCCompletePath::sortAndConvert() {
    coo.elms.resize(pPpr->E);
    csc.x_cpu.resize(pPpr->E);
    //csc.xPtr_cpu.resize(pPpr->V+1);
    csc.neightSize_cpu.resize(pPpr->V);

    for (int i = 0; i < pPpr->E; i++) {
        coo.elms[i]={pPpr->x[i],pPpr->y[i]};
    }
    std::sort(coo.elms.begin(),coo.elms.end());

    cooToCsc();
}

void MCCompletePath::cooToCsc() {
    int previousY;

    if (pPpr->E == 0)
        return;

    previousY = 0;
    csc.xPtr_cpu.push_back(0);

    for (int i = 0; i < pPpr->E; i++) {
        csc.x_cpu[i]=coo.elms[i].x;
        while (coo.elms[i].y != previousY) {
            csc.xPtr_cpu.push_back(i);
            previousY++;
        }
    }

    while (csc.xPtr_cpu.size() < pPpr->V + 1) {
        csc.xPtr_cpu.push_back(pPpr->E);
    }

    for (int i = 0; i < pPpr->V; i++) {
        //print xPtr_cpu
        //printf("xPtr_cpu %d ",csc.xPtr_cpu[i]);
        csc.neightSize_cpu[i] = csc.xPtr_cpu[i+1] - csc.xPtr_cpu[i];
        //print
        //std::cout << "Neighborhood size: " << csc.neightSize_cpu[i] << std::endl;
    }
}

void MCCompletePath::initQuasiRandom() {
    curandDirectionVectors32_t *hostVectors32;
    unsigned int * hostScrambleConstants32;

    curandStateScrambledSobol32 *devSobol32States;
    unsigned int * devDirectionVectors32;
    unsigned int * devScrambleConstants32;

    /* Get pointers to the 64 bit scrambled direction vectors and constants*/
    curandErrCheck(curandGetDirectionVectors32( &hostVectors32,CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    curandErrCheck(curandGetScrambleConstants32( &hostScrambleConstants32));


    /* Allocate memory for 1 states per thread (x, y, z), each state to get a unique dimension */
    errCheck(cudaMalloc((void **)&devSobol32States,walkers * sizeof(curandStateScrambledSobol32)));
    /* Allocate memory and copy the set of vectors per thread to the device */
    errCheck(cudaMalloc((void **)&(devDirectionVectors32), walkers * VECTOR_SIZE * sizeof(unsigned int)));

    errCheck(cudaMemcpy(devDirectionVectors32, hostVectors32,walkers * VECTOR_SIZE * sizeof(unsigned int),cudaMemcpyHostToDevice));

    /* Allocate memory and copy 3 scramble constants (one costant per dimension)
       per thread to the device */
    errCheck(cudaMalloc((void **)&(devScrambleConstants32),walkers * sizeof(unsigned int)));

    errCheck(cudaMemcpy(devScrambleConstants32, hostScrambleConstants32,walkers * sizeof(unsigned int),cudaMemcpyHostToDevice));

    /* Initialize the states */

    setup_kernel<<<pPpr->B, pPpr->T>>>(devDirectionVectors32,
                                       devScrambleConstants32,
                                       devSobol32States);
    errCheck(cudaPeekAtLastError());
    errCheck(cudaDeviceSynchronize());
}


