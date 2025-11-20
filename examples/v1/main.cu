#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

void f0(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

__global__ void k1(const float* a, const float* b, float* c, int n, int p) {
    int i0 = blockIdx.y * blockDim.y + threadIdx.y;
    int j0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= n || j0 >= n) return;
    float s = 0.0f;
    for (int t = 0; t < n; ++t) {
        float x = a[i0 * n + t];
        int j1 = (j0 * p) % n;
        float y = b[t * n + j1];
        s += x * y;
    }
    c[i0 * n + j0] = s;
}

int main() {
    const int n = 64;
    size_t bytes = n * n * sizeof(float);

    float* h_a = (float*)std::malloc(bytes);
    float* h_b = (float*)std::malloc(bytes);
    float* h_c = (float*)std::malloc(bytes);
    float* h_r = (float*)std::malloc(bytes);

    if (!h_a || !h_b || !h_c || !h_r) {
        std::fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int i = 0; i < n * n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = float(i % 13);
        h_c[i] = 0.0f;
        h_r[i] = 0.0f;
    }

    f0(h_r, h_a, h_b, n);

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_c, 0, bytes));

    dim3 blk(16, 16);
    dim3 grd((n + blk.x - 1) / blk.x, (n + blk.y - 1) / blk.y);

    int p = 8;
    k1<<<grd, blk>>>(d_a, d_b, d_c, n, p);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n * n; ++i) {
        float d = std::fabs(h_c[i] - h_r[i]);
        if (d > 1e-2f) {
            std::fprintf(stderr, "mismatch at %d: %f vs %f\n", i, h_c[i], h_r[i]);
            ok = false;
            break;
        }
    }

    std::printf("result %s\n", ok ? "ok" : "bad");

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::printf("%8.2f ", h_c[i * n + j]);
        }
        std::printf("\n");
    }

    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_r);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return 0;
}
