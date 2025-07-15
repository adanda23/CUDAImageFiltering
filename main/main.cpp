#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <string>

// Defining grayscale filter
void grayscaleCUDA(unsigned char* input, unsigned char* output, int width, int height);
void sepiaCUDA(unsigned char* input, unsigned char* output, int width, int height);


int main() {
    cv::Mat input = cv::imread("example.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return 1;
    }

    // Times 3 due to RGB channels
    int imgSize = input.cols * input.rows * 3;
    unsigned char *d_input, *d_output;

    //Freeing space for the size of the image and then copying the data into the GPU's memory
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, input.data, imgSize, cudaMemcpyHostToDevice);

    std::string filter;
    std::cout << "Select a filter: ";
    std::cin >> filter;

    if (filter == "grayscale")
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start recording time
        cudaEventRecord(start);

        grayscaleCUDA(d_input, d_output, input.cols, input.rows);

        // Stop recording time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    if (filter == "sepia")
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start recording time
        cudaEventRecord(start);

        sepiaCUDA(d_input, d_output, input.cols, input.rows);

        // Stop recording time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
        
    //Write back to host
    cv::Mat output(input.size(), input.type());
    cudaMemcpy(output.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    cv::imwrite("output.jpg", output);

    //Freeing the memory
    cudaFree(d_input);
    cudaFree(d_output);


    return 0;
}
