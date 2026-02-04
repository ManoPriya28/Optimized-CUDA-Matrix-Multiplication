# Optimized-CUDA-Matrix-Multiplication
High-performance CUDA implementation achieving 4x speedup via Shared Memory Tiling
# High-Performance CUDA Matrix Multiplication

![unnamed](https://github.com/user-attachments/assets/bcc149d7-5393-4e34-8b13-9a21d13f28a8)


## 🚀 Project Overview
This project implements a highly optimized Matrix Multiplication kernel using **CUDA C++**. The goal was to maximize GFLOPS on an NVIDIA Tesla T4 GPU by managing memory access patterns and utilizing Shared Memory (L1 Cache).

We successfully achieved a **4x speedup** compared to the naive implementation.

## 📊 Performance Results
Hardware: **NVIDIA Tesla T4**
Matrix Size: **2048 x 2048**

| Implementation | Block Size | Execution Time | Performance | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Naive (Global Mem)** | N/A | 75.03 ms | 228 GFLOPS | 1.0x |
| **Tiled (Shared Mem)** | 16x16 | 46.25 ms | 371 GFLOPS | 1.6x |
| **Optimized Tiled** | **32x32** | **18.45 ms** | **931 GFLOPS** | **4.0x** |

## 🛠️ Key Optimizations
1.  **Shared Memory Tiling:** Implemented a blocked algorithm to load data chunks into on-chip Shared Memory, drastically reducing Global Memory bandwidth pressure.
2.  **Block Size Tuning:** Experimented with 8x8, 16x16, and 32x32 block sizes. Found that **32x32** provided the optimal balance of register pressure and occupancy for the Tesla T4 architecture.
3.  **Memory Coalescing:** Structured threads to access global memory in contiguous patterns to maximize bus utilization.

## 💻 How to Run
Since this requires an NVIDIA GPU, the easiest way to reproduce these results is via Google Colab.

1.  Clone this repository.
2.  Compile using `nvcc`:
```bash
nvcc -o matmul matrix_mul_experiment.cu -arch=sm_75
