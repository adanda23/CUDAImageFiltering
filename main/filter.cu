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

// box blur filter
__global__ void boxBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = (y * width + x) * 3;
        float r = 0.0f, g = 0.0f, b = 0.0f;
        int count = 0;

        // 3x3 kernel
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int ni = (ny * width + nx) * 3;
                    r += input[ni];
                    g += input[ni + 1];
                    b += input[ni + 2];
                    count++;
                }
            }
        }

        r /= count;
        g /= count;
        b /= count;

        output[i]     = (r > 255.0f) ? 255 : (unsigned char)r;
        output[i + 1] = (g > 255.0f) ? 255 : (unsigned char)g;
        output[i + 2] = (b > 255.0f) ? 255 : (unsigned char)b;
    }
}


void boxBlurCUDA(unsigned char* input, unsigned char* output, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    boxBlurKernel<<<blocks, threads>>>(input, output, width, height);
}

// invert colors filter
__global__ void invertKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = (y * width + x) * 3;
        output[i] = 255 - input[i];         
        output[i + 1] = 255 - input[i + 1]; 
        output[i + 2] = 255 - input[i + 2]; 
    }
}

void invertCUDA(unsigned char* input, unsigned char* output, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    invertKernel<<<blocks, threads>>>(input, output, width, height);
}
