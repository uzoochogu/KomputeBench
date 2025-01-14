#include <stdio.h>
#define N 64
#define TPB 32

__device__ float scale(int i, int n) { return ((float)i) / (n - 1); }

__device__ float distance(float x1, float x2) {
  return sqrt((x1 - x2) * (x1 - x2));
}

__global__ void distanceKernel(float *d_out, float ref, int len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float x = scale(i, len);
  d_out[i] = distance(x, ref);
  printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

int main() {
  const float ref = 0.5f;

  // Declare a pointer for an array of floats
  float *d_out = 0;

  // Allocate device memory to store the output array
  cudaMalloc(&d_out, N * sizeof(float));

  // Launch Kernel to compute and store distance values
  distanceKernel<<<(N + TPB - 1) / TPB, TPB>>>(d_out, ref, N);
  // cudaDeviceSynchronize();
  float h_out[N];
  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    printf("i = %2d: dist from %f is %f.\n", i, ref, h_out[i]);
  }

  cudaFree(d_out);

  return 0;
}