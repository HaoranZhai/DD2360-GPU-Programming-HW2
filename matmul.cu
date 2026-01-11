#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                           \
    do {                                           \
        cudaError_t err = (call);                  \
        if (err != cudaSuccess) {                  \
            fprintf(stderr,                        \
                    "CUDA Error at %s:%d: %s\n",   \
                    __FILE__, __LINE__,            \
                    cudaGetErrorString(err));      \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)

// =======================
// Naive GEMM kernel
// =======================
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// =======================
// Tiled GEMM kernel
// =======================
template <int TILE>
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int m, int n, int k)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aRow = row;
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        int bCol = col;

        if (aRow < m && aCol < k) {
            As[threadIdx.y][threadIdx.x] = A[aRow * k + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < k && bCol < n) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * n + bCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// =======================
// Host helpers
// =======================
void fill_random(float* mat, int size)
{
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

float max_abs_error(const float* ref, const float* test, int size)
{
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

void cpu_matmul(const float* A, const float* B, float* C,
                int m, int n, int k)
{
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += A[row * k + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// =======================
// Benchmark for one (m,n,k)
// =======================
void run_case(int m, int n, int k)
{
    std::printf("\n==============================\n");
    std::printf("Matrix dimensions: A(%d x %d) * B(%d x %d)\n",
                m, k, k, n);

    size_t sizeA = static_cast<size_t>(m) * k * sizeof(float);
    size_t sizeB = static_cast<size_t>(k) * n * sizeof(float);
    size_t sizeC = static_cast<size_t>(m) * n * sizeof(float);

    float* A     = static_cast<float*>(std::malloc(sizeA));
    float* B     = static_cast<float*>(std::malloc(sizeB));
    float* C_cpu = static_cast<float*>(std::malloc(sizeC));
    float* C_gpu = static_cast<float*>(std::malloc(sizeC));
    float* C_tmp = static_cast<float*>(std::malloc(sizeC));

    if (!A || !B || !C_cpu || !C_gpu || !C_tmp) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    fill_random(A, m * k);
    fill_random(B, k * n);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice));

    // ---- CPU reference ----
    std::printf("Running CPU reference...\n");
    std::clock_t cpu_start = std::clock();
    cpu_matmul(A, B, C_cpu, m, n, k);
    std::clock_t cpu_end = std::clock();
    double cpu_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::printf("CPU done, time = %.3f ms\n", cpu_ms);

    // ---- Naive kernel ----
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(dA, dB, dC, m, n, k);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float naive_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(C_gpu, dC, sizeC, cudaMemcpyDeviceToHost));

    float err_naive = max_abs_error(C_cpu, C_gpu, m * n);
    std::printf("Naive kernel done, max error = %.8e, time = %.3f ms\n",
                err_naive, naive_ms);

    // ---- Tiled kernels: 8, 16, 32 ----
    int tileSizes[3] = {8, 16, 32};

    for (int ti = 0; ti < 3; ++ti) {
        int T = tileSizes[ti];
        dim3 blockT(T, T);
        dim3 gridT((n + T - 1) / T, (m + T - 1) / T);

        CHECK_CUDA(cudaMemset(dC, 0, sizeC));
        CHECK_CUDA(cudaEventRecord(start));

        switch (T) {
            case 8:
                matmul_tiled<8><<<gridT, blockT>>>(dA, dB, dC, m, n, k);
                break;
            case 16:
                matmul_tiled<16><<<gridT, blockT>>>(dA, dB, dC, m, n, k);
                break;
            case 32:
                matmul_tiled<32><<<gridT, blockT>>>(dA, dB, dC, m, n, k);
                break;
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float tiled_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&tiled_ms, start, stop));
        CHECK_CUDA(cudaMemcpy(C_tmp, dC, sizeC, cudaMemcpyDeviceToHost));

        float err_tiled = max_abs_error(C_cpu, C_tmp, m * n);
        std::printf("Tiled kernel (%2dx%2d): max error = %.8e, time = %.3f ms\n",
                    T, T, err_tiled, tiled_ms);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    std::free(A);
    std::free(B);
    std::free(C_cpu);
    std::free(C_gpu);
    std::free(C_tmp);
}

// =======================
// Main
// =======================
int main()
{
    std::printf("===== CUDA Homework 2 Q3: Tiled Matrix Multiplication =====\n");

    // For Q5 we use 8 different square sizes from small to larger.
    // You can change this list if you want other sizes.
    int sizes[] = {256, 384, 512, 640, 768, 896, 1024, 1152};
    int numSizes = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));

    std::printf("\n=== Question 5: multiple matrix sizes ===\n");

    for (int i = 0; i < numSizes; ++i) {
        int n = sizes[i];
        std::printf("\n[Q5] Running case with n = %d\n", n);
        run_case(n, n, n); // A(n x n) * B(n x n)
    }

    return 0;
}
