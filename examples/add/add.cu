// add.cu
#include <stdio.h>

__global__ void addKernel(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int a[5] = {1,2,3,4,5};
    int b[5] = {10,20,30,40,50};
    int c[5];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 5*sizeof(int));
    cudaMalloc(&d_b, 5*sizeof(int));
    cudaMalloc(&d_c, 5*sizeof(int));

    cudaMemcpy(d_a, a, 5*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 5*sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1,5>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, 5*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result = [%d %d %d %d %d]\n", c[0], c[1], c[2], c[3], c[4]);

    return 0;
}
