#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256   // threads per block

// ========================= CPU reduction =========================
float cpu_reduce(const float* input, int N) {
    double sum = 0.0;   // use double for better accuracy on CPU
    for (int i = 0; i < N; ++i) {
        sum += input[i];
    }
    return static_cast<float>(sum);
}

// ========================= GPU kernel =========================
// Each block reduces a chunk of the array into shared memory.
// At the end, thread 0 of each block atomically adds its partial
// sum to a single global result.
__global__ void reduce_kernel(const float* __restrict__ input,
                              float* __restrict__ result,
                              int N) {
    extern __shared__ float sdata[];  // size = BLOCK_SIZE * sizeof(float)

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    // load up to 2 elements per thread from global memory
    if (idx < (unsigned int)N) {
        sum += input[idx];
    }
    if (idx + blockDim.x < (unsigned int)N) {
        sum += input[idx + blockDim.x];
    }

    // store to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // parallel reduction in shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // add block result to global result
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ========================= Helper: init data =========================
void init_input(float* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;  // [0,1)
    }
}

// ========================= Main program =========================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Error: N must be positive\n");
        return 1;
    }

    srand(0);  // fixed seed for reproducibility

    // host memory
    float* h_input = (float*)malloc(N * sizeof(float));
    float  h_cpu   = 0.0f;
    float  h_gpu   = 0.0f;

    init_input(h_input, N);

    // -------- CPU timing --------
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);

    h_cpu = cpu_reduce(h_input, N);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float ms_cpu = 0.0f;
    cudaEventElapsedTime(&ms_cpu, start_cpu, stop_cpu);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);

    // -------- Device memory --------
    float *d_input = nullptr, *d_result = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // -------- Kernel launch config --------
    int grid = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    size_t shmem_size = BLOCK_SIZE * sizeof(float);

    // -------- GPU timing --------
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    reduce_kernel<<<grid, BLOCK_SIZE, shmem_size>>>(d_input, d_result, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float ms_gpu = 0.0f;
    cudaEventElapsedTime(&ms_gpu, start_gpu, stop_gpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cudaMemcpy(&h_gpu, d_result, sizeof(float),
               cudaMemcpyDeviceToHost);

    float diff = fabsf(h_cpu - h_gpu);

    // header + single result line (方便脚本抓最后一行)
    printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);
    printf("%12s %12s %12s %15s\n", "N", "CPU(ms)", "GPU(ms)", "abs diff");
    printf("%12d %12.4f %12.4f %15.6f\n",
           N, ms_cpu, ms_gpu, diff);

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
