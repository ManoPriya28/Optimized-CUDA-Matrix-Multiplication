#include <iostream>
#include <cuda_runtime.h>

// 1. Naive Kernel
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

// 2. Main Function
int main() {
    int M = 1024, K = 1024, N = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate Host Memory
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    // Initialize
    for(int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K*N; i++) h_B[i] = 1.0f;

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // Copy Data
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // Run Kernel
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulNaive<<<blocks, threads>>>(d_C, d_A, d_B, M, K, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Matrix Size: %dx%d\n", M, N);
    printf("Time Taken: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", (2.0f * M * N * K) / (milliseconds * 1e6));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
