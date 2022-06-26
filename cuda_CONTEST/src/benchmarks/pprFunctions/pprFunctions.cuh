#pragma once
#include "cuda_runtime_api.h"
#include <cuda_fp16.h>
template<typename T1, typename T2> __global__ void dot_product_kernel_math(const  T1 * __restrict__ x,
                                                                           const   T2 * __restrict__ y,
                                                                           T2  * __restrict__ dot,
                                                                           T2 dampingFract,
                                                                           unsigned int n){


    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;

    extern __shared__ T2 cache[];

    T2 temp = 0.0;

    while(index < n){
        // temp += x[index]*y[index];
        temp = __fmaf_rd(x[index], y[index], temp);
        index += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    // reduction
    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    unsigned int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] = __fadd_rd(cache[threadIdx.x], cache[threadIdx.x + i]);
        }
        __syncthreads();
        i /= 2;
    }

    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(dot, dampingFract*cache[0]);
    }
}

template<typename T1, typename T2> __global__ void dot_product_kernel(T1*x, T2 *y, T2 *dot, T2 dampingFract, unsigned int n){

    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;

    extern __shared__ T2 cache[];

    T2 temp = 0.0;
    while(index < n){
        temp += x[index]*y[index];

        index += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }


    if(threadIdx.x == 0){
        atomicAdd(dot, dampingFract* cache[0]);
    }
}


// FOR THIS KERNEL NUMBER OF THREADS SHOULD BE EQUAL TO DANGLING_SIZE, THIS KERNEL WORKS WITH ONLY INDEXES OF DANGLING!!!
template<typename T1, typename T2> __global__ void dangling_kernel(T1*x, T2 *y, T2 *dot, T2 dampingFract, unsigned int n){

    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;

    extern __shared__ T2 cache[];

    T2 temp = 0.0;
    while(index < n){

        temp = __fadd_rd(temp, y[x[index]]);
        index += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] = __fadd_rd(cache[threadIdx.x], cache[threadIdx.x + i]);
        }
        __syncthreads();
        i /= 2;
    }


    if(threadIdx.x == 0){
        atomicAdd(dot, dampingFract * cache[0]);
    }
}

template<typename T> __global__ void euclidean_kernel_math(const T * __restrict__ x, const T * __restrict__ y, T * __restrict__ result, unsigned int n)
{
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    extern __shared__ T cache1[];

    T temp = 0.0;
    while (index < n) {
        temp =  __fadd_rd(temp, (
                                  __fmul_rd(
                                          __fsub_rd(x[index],y[index]),
                                          __fsub_rd(x[index],y[index])
                                  )
                          )
        );

        index += stride;
    }

    cache1[threadIdx.x] = temp;

    __syncthreads();

    // reduction
    size_t i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cache1[threadIdx.x] = __fadd_rd(cache1[threadIdx.x], cache1[threadIdx.x + i]);
        }
        __syncthreads();
        i /= 2;
    }


    if (threadIdx.x == 0) {
        atomicAdd(result, cache1[0]); //
    }

}

template <typename T> __global__ void vectorScalarAddAndIncrement( T scalar, T *vector, int array_len, int index, T amount){
    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        if(i == index) vector[i] = vector[i] + scalar + amount;
        else vector[i] += scalar;
        i += gridSize;
    }
}

template<typename T1, typename T2> __global__ void compute_dangling_factor_gpu( T1 *dangling,  T2* pr, T2 *result, int V){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += blockDim.x * gridDim.x){
        T2 val = (T2) dangling[i] * pr[i];
        atomicAdd(result, val);
    }
}

template<typename T> __global__ void compute_square_error_gpu_math(
        const  T * __restrict__ old,
        const  T * __restrict__ newVector,
        T * __restrict__  result, int V){

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += blockDim.x * gridDim.x) {
        atomicAdd(result,
                  __fmul_rd(
                          __fsub_rd(old[i], newVector[i]),
                          __fsub_rd(old[i], newVector[i])));
    }

}

template<typename T> __global__ void compute_square_error_gpu( T *old,  T *newVector, T *result, int V){
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += blockDim.x * gridDim.x) {
        atomicAdd(result, (T)((old[i] - newVector[i]) * (old[i] - newVector[i])));
    }
}

template<typename T> __global__ void vectorScalarMul_math( T scalar, T *vector, int array_len){
    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] = __fmul_rd(vector[i], scalar);
        i += gridSize;
    }
}

template<typename T> __global__ void vectorScalarMul( T scalar, T *vector, int array_len){
    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] *= scalar;
        i += gridSize;
    }
}

template <typename T> __global__ void vectorScalarAdd( T scalar, T *vector, int array_len){
    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        vector[i] += scalar;
        i += gridSize;
    }
}

template <typename T> __global__ void vectorScalarAddAndIncrement_math(
        T scalar,
        T *vector,
        int array_len, int index, T amount){
    size_t tid = threadIdx.x,
            gridSize = blockDim.x * gridDim.x,
            i = blockIdx.x * blockDim.x + tid;

    while (i < array_len) {
        if(i == index) vector[i] = __fadd_rd(__fadd_rd(vector[i], scalar),amount);
        else vector[i] = __fadd_rd(vector[i], scalar);
        i += gridSize;
    }
}

template <typename T1, typename T2> __global__ void cooSPMV_math(
        const T1 *  __restrict__  x_gpu,
        const T1 *  __restrict__  y_gpu,
        const T2 * __restrict__ val_gpu,
        const T1 E,
        const T2 * __restrict__ pr_old,
        T2 * __restrict__ pr_temp)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;
    if (index < E){
        atomicAdd(&pr_temp[x_gpu[index]], val_gpu[index]*pr_old[y_gpu[index]]);
        index += gridSize;
    }
}

template <typename T1, typename T2> __global__ void cooSPMV(
        const T1 *  __restrict__  x_gpu,
        const T1 *  __restrict__  y_gpu,
        const T2 * __restrict__ val_gpu,
        const T1 E,
        const T2 * __restrict__ pr_old,
        T2 * __restrict__ pr_temp)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;
    if (index < E){
        atomicAdd(&pr_temp[x_gpu[index]], val_gpu[index]*pr_old[y_gpu[index]]);
        index += gridSize;
    }
}

template<typename T> __global__ void init_vector(T* v, int size, T value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        v[i] = value;
        i += gridDim.x * blockDim.x;
    }
}

template<typename T> __global__ void copy_vector(T* dest, T* source, int size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        dest[i] = source[i];
        i += gridDim.x * blockDim.x;
    }
}

template<typename T1, typename T2> __global__ void cast_vector(T1* dest, T2* source, int size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride =  gridDim.x * blockDim.x;

#pragma unroll
    while (i < 3566907) {
        dest[i] = source[i];
        i += stride;
    }


}

template <typename T> __global__ void copy_vector_and_increment_ppr(T* dest, T* source, int size, int index, T val){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        dest[i] = source[i];
        if(i == index) dest[i] += val;
        i += gridDim.x * blockDim.x;
    }
}

template<typename T> __global__ void incremetBy1(T *arr, int idx,T value) {
    arr[idx] += value;
}

// Parallel SpMV with one Warp per Row, NEEDS 1024 THREADS
template <typename T> __global__ void parallel_spmv_csr_math(const T * __restrict__ V,
                                                             const int * __restrict__ y,
                                                             const int * __restrict__ rowPtr,
                                                             const T * __restrict__ ppr,
                                                             T * out,
                                                             int N){
    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t warp_id = thread_id / 32;
    size_t lane_id = thread_id % 32;

    size_t row = warp_id;

    if(row < N){
        size_t begin_index = rowPtr[row];
        size_t end_index = rowPtr[row+1];

        T thread_sum = 0.0;
        for(size_t i = begin_index + lane_id; i < end_index; i+=32)
            thread_sum = __fadd_rd(thread_sum, __fmul_rd(V[i],ppr[y[i]]));

        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum,16);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum,8);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum,4);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum,2);
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum,1);

        if(lane_id == 0)
            out[row] = thread_sum;

    }
}

template<typename T> __global__ void compute_aikten_x_math(
        T* x,
        const  T* __restrict__ xMinus1,
        const  T* __restrict__ xMinus2, int size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        x[i] =  __fsub_rd(
                xMinus2[i],
                (__fdividef(
                        (T)(__fmul_rd(
                                (__fsub_rd(xMinus1[i],xMinus2[i])),
                                (__fsub_rd(xMinus1[i],xMinus2[i]))
                        )),

                        (__fsub_rd(x[i],2*xMinus1[i]) + xMinus2[i])
                )));

        i += gridDim.x * blockDim.x;
    }
}
