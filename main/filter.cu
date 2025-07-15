#include <cuda_runtime.h>

// grayscale filter
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

// sepia filter
__global__ void sepiaKernal(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int i = (y * width + x) * 3;
        unsigned char r = input[i];
        unsigned char g = input[i + 1];
        unsigned char b = input[i + 2];
        float updatedR = 0.393*r + 0.769*g + 0.189*b;
        float updatedG = 0.349*r + 0.686*g + 0.168*b;
        float updatedB = 0.272*r + 0.534*g + 0.131*b;
        //keeping them below 255
        output[i] = (updatedR > 255.0f) ? 255 : (unsigned char)updatedR;
        output[i + 1] = (updatedG > 255.0f) ? 255 : (unsigned char)updatedG;
        output[i + 2] = (updatedB > 255.0f) ? 255 : (unsigned char)updatedB;
    }
}

void sepiaCUDA(unsigned char* input, unsigned char* output, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    sepiaKernal<<<blocks, threads>>>(input, output, width, height);
}
