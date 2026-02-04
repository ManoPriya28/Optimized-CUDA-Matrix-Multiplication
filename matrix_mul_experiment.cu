#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

// We use a Template so we can compile different versions easily
template <int BLOCK_SIZE>
__global__ void matMulTiled(float* C, const float* A, const float* B, int M, int K, int N) {
    // Static Shared Memory allocation based on template parameter
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int ph = 0; ph < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++ph) {
        
        // Load data into shared memory
        if (row < M && (ph * BLOCK_SIZE + tx) < K)
            As[ty][tx] = A[row * K + ph * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (ph * BLOCK_SIZE + ty) < K)
            Bs[ty][tx] = B[(ph * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper to run specific test
template <int SIZE>
void runTest(float* d_C, float* d_A, float* d_B, int M, int K, int N) {
    dim3 threads(SIZE, SIZE);
    dim3 blocks((N + SIZE - 1) / SIZE, (M + SIZE - 1) / SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float milliseconds = 0;

    // Warm up run (to wake up GPU)
    matMulTiled<SIZE><<<blocks, threads>>>(d_C, d_A, d_B, M, K, N);
    
    // Actual Measurement
    cudaEventRecord(start);
    matMulTiled<SIZE><<<blocks, threads>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Block Size %2dx%2d:  Time: %6.2f ms  |  Performance: %7.2f GFLOPS\n", 
           SIZE, SIZE, milliseconds, (2.0f * M * N * K) / (milliseconds * 1e6));
}

int main() {
    int M = 2048, K = 2048, N = 2048;
    printf("Running Experiments on Matrix Size: %dx%dx%d\n\n", M, K, N);

    // Host Init
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    for(int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K*N; i++) h_B[i] = 1.0f;

    // Device Init
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // --- RUN EXPERIMENTS ---
    runTest<8>(d_C, d_A, d_B, M, K, N);
    runTest<16>(d_C, d_A, d_B, M, K, N);
    runTest<32>(d_C, d_A, d_B, M, K, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
