#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 32  // We keep 32 because you proved it was the best!

__global__ void matMulVectorized(float* C, const float* A, const float* B, int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // We loop over tiles
    for (int ph = 0; ph < K / TILE_WIDTH; ++ph) {
        
        // --- MEMORY ACCESS OPTIMIZATION ---
        // Instead of verifying every single index with "if" statements (which is slow),
        // We assume matrix size is a multiple of 32 (standard in HPC).
        // This removes "Branch Divergence".
        
        As[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];
        Bs[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main() {
    // Dimensions must be multiples of 32 for this specific optimized kernel
    int M = 2048, K = 2048, N = 2048; 
    
    // Host Init
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N]; // Host result

    for(int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K*N; i++) h_B[i] = 1.0f;

    // Device Init
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(N / TILE_WIDTH, M / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float milliseconds = 0;

    // Warmup
    matMulVectorized<<<blocks, threads>>>(d_C, d_A, d_B, M, K, N);

    // Measure
    cudaEventRecord(start);
    matMulVectorized<<<blocks, threads>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Vectorized/Streamlined Kernel (32x32):\n");
    printf("Time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", (2.0f * M * N * K) / (milliseconds * 1e6));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
