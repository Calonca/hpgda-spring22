#pragma once
#include "cuda_runtime_api.h"
// T1 -> int
// T2 -> float/double

#define WARP_SIZE 32

template <typename T1, typename T2> __global__ void cooSPMV_serial(
        const T1 *  __restrict__  x_gpu,
        const T1 *  __restrict__  y_gpu,
        const T2 * __restrict__ val_gpu,
        const T1 E,
        const T2 * __restrict__ pr_old,
        T2 * __restrict__ pr_temp)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < E){
        atomicAdd(&pr_temp[x_gpu[index]], val_gpu[index]*pr_old[y_gpu[index]]);

    }
}



template <typename T1, typename T2> __device__ void segreduce_block(const T1 * __restrict__ idx, T2 * __restrict__ val)
{
    T2 left = 0;
    if( threadIdx.x >=   1 && idx[threadIdx.x] == idx[threadIdx.x -   1] ) {
        left = val[threadIdx.x -   1];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >=   2 && idx[threadIdx.x] == idx[threadIdx.x -   2] ) {
        left = val[threadIdx.x -   2];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >=   4 && idx[threadIdx.x] == idx[threadIdx.x -   4] ) {
        left = val[threadIdx.x -   4];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >=   8 && idx[threadIdx.x] == idx[threadIdx.x -   8] ) {
        left = val[threadIdx.x -   8];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >=  16 && idx[threadIdx.x] == idx[threadIdx.x -  16] ) {
        left = val[threadIdx.x -  16];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >=  32 && idx[threadIdx.x] == idx[threadIdx.x -  32] ) {
        left = val[threadIdx.x -  32];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >=  64 && idx[threadIdx.x] == idx[threadIdx.x -  64] ) {
        left = val[threadIdx.x -  64];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128] ) {
        left = val[threadIdx.x - 128];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
    if( threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256] ) {
        left = val[threadIdx.x - 256];
    }
    __syncthreads();
    val[threadIdx.x] = __fadd_rd(val[threadIdx.x], left);
    left = 0;
    __syncthreads();
}


//////////////////////////////////////////////////////////////////////////////
// COO SpMV kernel which flattens data irregularity (segmented reduction)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_coo_flat
//   The input coo_matrix must be sorted by row.  Columns within each row
//   may appear in any order and duplicate entries are also acceptable.
//   This sorted COO format is easily obtained by expanding the row pointer
//   of a CSR matrix (csr.Ap) into proper row indices and then copying
//   the arrays containing the CSR column indices (csr.Aj) and nonzero values
//   (csr.Ax) verbatim.  A segmented reduction is used to compute the per-row
//   sums.
//
// spmv_coo_flat_tex
//   Same as spmv_coo_flat, except that the texture cache is
//   used for accessing the x vector.
//


// spmv_coo_flat_kernel
//
// In this kernel each warp processes an interval of the nonzero values.
// For example, if the matrix contains 128 nonzero values and there are
// two warps and interval_size is 64, then the first warp (warp_id == 0)
// will process the first set of 64 values (interval [0, 64)) and the
// second warp will process // the second set of 64 values
// (interval [64, 128)).  Note that the  number of nonzeros is not always
// a multiple of 32 (the warp size) or 32 * the number of active warps,
// so the last active warp will not always process a "full" interval of
// interval_size.
//
// The first thread in each warp (thread_lane == 0) has a special role:
// it is responsible for keeping track of the "carry" values from one
// iteration to the next.  The carry values consist of the row index and
// partial sum from the previous batch of 32 elements.  In the example
// mentioned before with two warps and 128 nonzero elements, the first
// warp iterates twice and looks at the carry of the first iteration to
// decide whether to include this partial sum into the current batch.
// Specifically, if a row extends over a 32-element boundary, then the
// partial sum is carried over into the new 32-element batch.  If,
// on the other hand, the _last_ row index of the previous batch (the carry)
// differs from the _first_ row index of the current batch (the row
// read by the thread with thread_lane == 0), then the partial sum
// is written out to memory.
//
// Each warp iterates over its interval, processing 32 elements at a time.
// For each batch of 32 elements, the warp does the following
//  1) Fetch the row index, column index, and value for a matrix entry.  These
//     values are loaded from x[n], y[n], and val[n] respectively.
//     The row entry is stored in the shared memory array idx.
//  2) Fetch the corresponding entry from the input vector.  Specifically, for a
//     nonzero entry (i,j) in the matrix, the thread must load the value x[j]
//     from memory.  We use the function fetch_x to control whether the texture
//     cache is used to load the value (UseCache == True) or whether a normal
//     global load is used (UseCache == False).
//  3) The matrix value A(i,j) (which was stored in V[n]) is multiplied by the
//     value x[j] and stored in the shared memory array val.
//  4) The first thread in the warp (thread_lane == 0) considers the "carry"
//     row index and either includes the carried sum in its own sum, or it
//     updates the output vector (res) with the carried sum.
//  5) With row indices in the shared array idx and sums in the shared array
//     val, the warp conducts a segmented scan.  The segmented scan operation
//     looks at the row entries for each thread (stored in idx) to see whether
//     two values belong to the same segment (segments correspond to matrix rows).
//     Consider the following example which consists of 3 segments
//     (note: this example uses a warp size of 16 instead of the usual 32)
//
//           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   # thread_lane
//     idx [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  # row indices
//     val [ 4, 6, 5, 0, 8, 3, 2, 8, 3, 1, 4, 9, 2, 5, 2, 4]  # A(i,j) * x(j)
//
//     After the segmented scan the result will be
//
//     val [ 4,10,15,15,23,26, 2,10,13,14, 4,13,15,20,22,26]  # A(i,j) * x(j)
//
//  6) After the warp computes the segmented scan operation
//     each thread except for the last (thread_lane == 31) looks
//     at the row index of the next thread (threadIdx.x + 1) to
//     see if the segment ends here, or continues into the
//     next thread.  The thread at the end of the segment writes
//     the sum into the output vector (res) at the corresponding row
//     index.
//  7) The last thread in each warp (thread_lane == 31) writes
//     its row index and partial sum into the designated spote in the
//     carry_idx and carry_val arrays.  The carry arrays are indexed
//     by warp_lane which is a number in [0, BLOCK_SIZE / 32).
//
//  These steps are repeated until the warp reaches the end of its interval.
//  The carry values at the end of each interval are written to arrays
//  temp_rows and temp_vals, which are processed by a second kernel.
//

// T1 -> int
// T2 -> float/double
template <typename T1, typename T2, unsigned int BLOCK_SIZE> __launch_bounds__(BLOCK_SIZE,1) __global__ void spmv_coo_flat_kernel(
        const T1 num_nonzeros,
        const T1 interval_size,
        const T1 * __restrict__ x,
        const T1 * __restrict__ y,
        const T2 * __restrict__ v,
        const T2 * __restrict__ ppr,
        T2 * __restrict__ res,
        T1 * __restrict__ temp_rows,
        T2 * __restrict__ temp_vals)
{
    __shared__ volatile T1 rows[48 *(BLOCK_SIZE/32)];
    __shared__ volatile T2 vals[BLOCK_SIZE];

    const T1 threadIndex   = blockDim.x * blockIdx.x + threadIdx.x;                         // global thread index
    const T1 threadWrapIndex = threadIdx.x & (WARP_SIZE-1);                                // thread index within the warp
    const T1 warp_id     = threadIndex   / WARP_SIZE;                                      // global warp index

    const T1 interval_begin = warp_id * interval_size;
    // warp's offset into x,y,val
    const T1 interval_end = min(interval_begin + interval_size, num_nonzeros); // end of warps's work

    const T1 idx = 16 * (threadIdx.x/32 + 1) + threadIdx.x;                               // thread's index into padded rows array

    rows[idx - 16] = -1;                                                                  // fill padding with invalid row index

    if(interval_begin >= interval_end)                                                    // warp has no work to do
        return;

    if (threadWrapIndex == 31)
    {
        // initialize the carry in values
        rows[idx] = x[interval_begin];
        vals[threadIdx.x] = 0.0;
    }

    for(T1 n = interval_begin + threadWrapIndex; n < interval_end; n += WARP_SIZE)
    {
        T1 row = x[n];                                     // row index (i)
        T2 val = __fmul_rn(v[n], ppr[ y[n] ]);            //  A(i,j) * ppr(j)

        if (threadWrapIndex == 0)
        {
            if(row == rows[idx + 31]) val = __fadd_rd(val, (vals[threadIdx.x + 31])); // row continues
            else res[rows[idx + 31]] = __fadd_rd(res[rows[idx + 31]], vals[threadIdx.x + 31]);  // row terminated
        }

        rows[idx] = row;
        vals[threadIdx.x] = val;

        if(row == rows[idx -  1]) {
            vals[threadIdx.x] = val = __fadd_rd(val, vals[threadIdx.x -  1]);
        }
        if(row == rows[idx -  2]) {
            vals[threadIdx.x] = val = __fadd_rd(val, vals[threadIdx.x -  2]);
        }
        if(row == rows[idx -  4]) {
            vals[threadIdx.x] = val = __fadd_rd(val, vals[threadIdx.x -  4]);
        }
        if(row == rows[idx -  8]) {
            vals[threadIdx.x] = val = __fadd_rd(val, vals[threadIdx.x -  8]);
        }
        if(row == rows[idx - 16]) {
            vals[threadIdx.x] = val = __fadd_rd(val, vals[threadIdx.x - 16]);
        }

        if(threadWrapIndex < 31 && row != rows[idx + 1])
            res[row] = __fadd_rd(res[row], vals[threadIdx.x]);
        // row terminated
    }

    if(threadWrapIndex == 31)
    {
        // write the carry out values
        temp_rows[warp_id] = rows[idx];
        temp_vals[warp_id] = vals[threadIdx.x];
    }
}


// The second level of the segmented reduction operation
template <typename T1, typename T2, unsigned int BLOCK_SIZE> __launch_bounds__(BLOCK_SIZE,1) __global__
void spmv_coo_reduce_update_kernel(const unsigned int num_warps,
                                   const T1 * __restrict__ temp_rows,
                                   const T2 * __restrict__ temp_vals,
                                   T2 * __restrict__ res)

{

    __shared__ T1 rows[BLOCK_SIZE + 1];
    __shared__ T2 vals[BLOCK_SIZE + 1];

    const T1 end = num_warps - (num_warps & (BLOCK_SIZE - 1));

    if (threadIdx.x == 0)
    {
        rows[blockDim.x] = -1;
        vals[blockDim.x] = 0.0;
    }

    __syncthreads();

    size_t i = threadIdx.x;

    while (i < end)
    {
        // do full blocks
        rows[threadIdx.x] = temp_rows[i];
        vals[threadIdx.x] = temp_vals[i];

        __syncthreads();

        segreduce_block(rows, vals);

        if (rows[threadIdx.x] != rows[threadIdx.x + 1])
            res[rows[threadIdx.x]] = __fadd_rd(res[rows[threadIdx.x]], vals[threadIdx.x]);

        __syncthreads();

        i += blockDim.x;
    }

    if (end < num_warps) {
        if (i < num_warps) {
            rows[threadIdx.x] = temp_rows[i];
            vals[threadIdx.x] = temp_vals[i];
        } else {
            rows[threadIdx.x] = -1;
            vals[threadIdx.x] =  0;
        }

        __syncthreads();

        segreduce_block(rows, vals);

        if (i < num_warps)
            if (rows[threadIdx.x] != rows[threadIdx.x + 1])
                res[rows[threadIdx.x]] = __fadd_rd(res[rows[threadIdx.x]], vals[threadIdx.x]);
    }
}


template <typename T1, typename T2, unsigned int BLOCK_SIZE> void __spmv_coo_flat(
        const T1 * __restrict__ x,
        const T1 * __restrict__ y,
        const T2 * __restrict__ val,
        const T2 * __restrict__ pr_old,
        T2 * __restrict__ res,
        T1 E,
        const T1 num_blocks,
        const T1 interval_size,
        const T1 tail,
        const T1 active_warps,
        T1 * __restrict__ temp_rows,
        T2 * __restrict__ temp_vals
)

{


    spmv_coo_flat_kernel<T1,T2, BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>(tail,
            interval_size,
            x,
            y,
            val,
            pr_old,
            res,
            temp_rows,
            temp_vals);


    spmv_coo_reduce_update_kernel<T1, T2, BLOCK_SIZE> <<<1, BLOCK_SIZE>>>(active_warps, temp_rows, temp_vals, res);

    cooSPMV_serial<T1, T2><<<1,1>>>(
            x + tail,
                    y + tail,
                    val + tail,
                    E - tail,
                    pr_old,
                    res);
}

