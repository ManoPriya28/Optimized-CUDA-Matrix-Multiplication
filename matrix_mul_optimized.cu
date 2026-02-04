#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

// --------------------------------------------------------
// 1. NAIVE KERNEL (Slow, Global Memory)
// --------------------------------------------------------
__global__ void matMulNaive(float* C, const float* A, const float* B, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --------------------------------------------------------
// 2. TILED KERNEL (Fast, Shared Memory)
// --------------------------------------------------------
__global__ void matMulTiled(float* C, const float* A, const float* B, int M, int K, int N) {
    // Shared memory for the tile (Fast L1 Cache)
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // Loop over the matrix in "Tiles"
    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {

        // 1. Cooperative loading: Each thread loads 1 element into shared memory
        if (row < M && (ph * TILE_WIDTH + tx) < K)
            Ads[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];
        else
            Ads[ty][tx] = 0.0f;

        if (col < N && (ph * TILE_WIDTH + ty) < K)
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        else
            Bds[ty][tx] = 0.0f;

        // 2. Wait for all threads to finish loading
        __syncthreads();

        // 3. Compute dot product using the FAST shared memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += Ads[ty][k] * Bds[k][tx];
        }

        // 4. Wait before loading the next tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    // Dimensions
    int M = 2048, K = 2048, N = 2048; // Larger size to see difference better
    size_t size = M * N * sizeof(float);

    printf("Matrix Size: %dx%d\n", M, N);

    // Host allocation
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    // Initialize
    for(int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K*N; i++) h_B[i] = 1.0f;

    // Device allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float milliseconds = 0;

    // Define Block/Grid
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // ------------------------------------------
    // RUN NAIVE
    // ------------------------------------------
    cudaEventRecord(start);
    matMulNaive<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nNaive Implementation:\n");
    printf("Time: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", (2.0f * M * N * K) / (milliseconds * 1e6));

    // ------------------------------------------
    // RUN TILED (SHARED MEMORY)
    // ------------------------------------------
    // Reset C
    cudaMemset(d_C, 0, size);

    cudaEventRecord(start);
    matMulTiled<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nTiled (Shared Mem) Implementation:\n");
    printf("Time: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", (2.0f * M * N * K) / (milliseconds * 1e6));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
