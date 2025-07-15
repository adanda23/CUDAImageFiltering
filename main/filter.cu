#include <cuda_runtime.h>

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = (y * width + x) * 3;
        unsigned char r = input[i];
        unsigned char g = input[i + 1];
        unsigned char b = input[i + 2];
        unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
        output[i] = output[i + 1] = output[i + 2] = gray;
    }
}

void grayscaleCUDA(unsigned char* input, unsigned char* output, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<blocks, threads>>>(input, output, width, height);
}

