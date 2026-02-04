#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    int M = 2048, K = 2048, N = 2048;
    
    // Host Data
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];
    for(int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K*N; i++) h_B[i] = 1.0f;

    // Device Data
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warm up
    float alpha = 1.0f; float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    // Measure
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Note: cuBLAS uses Column-Major storage, so we swap A and B parameters to fake Row-Major
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("NVIDIA cuBLAS Library Performance:\n");
    printf("Time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", (2.0f * M * N * K) / (milliseconds * 1e6));

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
