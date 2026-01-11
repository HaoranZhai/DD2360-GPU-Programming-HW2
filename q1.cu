#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define NUM_BINS 4096
#define SAT_LIMIT 127

// ========================= CPU Histogram =========================
void cpu_histogram(const unsigned int* input, unsigned int* hist, int N) {
    for (int i = 0; i < NUM_BINS; i++) hist[i] = 0;

    for (int i = 0; i < N; i++) {
        unsigned int v = input[i];
        if (hist[v] < SAT_LIMIT)
            hist[v]++;
    }
}

// ========================= GPU Kernel =========================
// Shared-memory histogram + saturation inside the same kernel
__global__
void histogram_kernel(const unsigned int* input, unsigned int* hist, int N) {

    __shared__ unsigned int local_hist[NUM_BINS];

    // Step 1: init shared memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    // Step 2: process input array
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N) {
        unsigned int v = input[global_id];
        atomicAdd(&local_hist[v], 1);
    }
    __syncthreads();

    // Step 3: write shared hist to global hist with atomicAdd and saturation
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        unsigned int val = local_hist[i];
        if (val > 0) {
            unsigned int old = atomicAdd(&hist[i], val);
            if (old + val > SAT_LIMIT)
                hist[i] = SAT_LIMIT;
        }
    }
}

// ========================= RNG Helper =========================
void generate_uniform(unsigned int* arr, int N) {
    for (int i = 0; i < N; i++)
        arr[i] = rand() % NUM_BINS;
}

void generate_normal(unsigned int* arr, int N) {
    double mu = NUM_BINS / 2.0;
    double sigma = NUM_BINS / 10.0;

    for (int i = 0; i < N; i++) {
        double x = mu + sigma * ((double)rand() / RAND_MAX * 2 - 1);
        if (x < 0) x = 0;
        if (x >= NUM_BINS) x = NUM_BINS - 1;
        arr[i] = (unsigned int)x;
    }
}

// ========================= Main Program =========================
int main() {
    // 固定种子，结果可重复
    srand(0);

    int lengths[4] = {1024, 10240, 102400, 1024000};

    for (int mode = 0; mode < 2; mode++) {
        printf("\n===================== Distribution: %s =====================\n",
               mode == 0 ? "Uniform" : "Normal");

        for (int t = 0; t < 4; t++) {

            int N = lengths[t];
            printf("\n---- N = %d ----\n", N);

            // Host alloc
            unsigned int* h_input = (unsigned int*)malloc(N * sizeof(unsigned int));
            unsigned int* h_cpu   = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
            unsigned int* h_gpu   = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

            // Init distribution
            if (mode == 0)
                generate_uniform(h_input, N);
            else
                generate_normal(h_input, N);

            // CPU timing
            cudaEvent_t start_cpu, stop_cpu;
            cudaEventCreate(&start_cpu);
            cudaEventCreate(&stop_cpu);
            cudaEventRecord(start_cpu);

            cpu_histogram(h_input, h_cpu, N);

            cudaEventRecord(stop_cpu);
            cudaEventSynchronize(stop_cpu);
            float ms_cpu;
            cudaEventElapsedTime(&ms_cpu, start_cpu, stop_cpu);

            // Device alloc
            unsigned int *d_input, *d_hist;
            cudaMalloc(&d_input, N * sizeof(unsigned int));
            cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));

            // Init device hist = 0
            cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

            // Copy input → GPU
            cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

            // Kernel launch config
            int block = 256;
            int grid  = (N + block - 1) / block;

            // GPU timing
            cudaEvent_t start_gpu, stop_gpu;
            cudaEventCreate(&start_gpu);
            cudaEventCreate(&stop_gpu);
            cudaEventRecord(start_gpu);

            histogram_kernel<<<grid, block>>>(d_input, d_hist, N);

            cudaEventRecord(stop_gpu);
            cudaEventSynchronize(stop_gpu);
            float ms_gpu;
            cudaEventElapsedTime(&ms_gpu, start_gpu, stop_gpu);

            // Copy back results
            cudaMemcpy(h_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            // correctness check
            int ok = 1;
            for (int i = 0; i < NUM_BINS; i++) {
                if (h_cpu[i] != h_gpu[i]) {
                    ok = 0;
                    printf("Mismatch at bin %d: CPU=%u GPU=%u\n", i, h_cpu[i], h_gpu[i]);
                    break;
                }
            }
            printf("Correctness: %s\n", ok ? "OK" : "FAIL");

            printf("CPU Time: %.4f ms\n", ms_cpu);
            printf("GPU Kernel Time: %.4f ms\n", ms_gpu);
            printf("blockDim = %d, gridDim = %d\n", block, grid);

            // 把 GPU 直方图写到文件，方便 Python 画图
            char fname[64];
            snprintf(fname, sizeof(fname),
                     "hist_%s_%d.txt",
                     mode == 0 ? "uniform" : "normal",
                     N);
            FILE *fh = fopen(fname, "w");
            if (fh) {
                for (int i = 0; i < NUM_BINS; i++) {
                    // CPU / GPU 一样，写 GPU 结果即可
                    fprintf(fh, "%u\n", h_gpu[i]);
                }
                fclose(fh);
            } else {
                printf("Failed to open file %s for writing\n", fname);
            }

            // Free
            free(h_input);
            free(h_cpu);
            free(h_gpu);
            cudaFree(d_input);
            cudaFree(d_hist);
        }
    }

    return 0;
}
