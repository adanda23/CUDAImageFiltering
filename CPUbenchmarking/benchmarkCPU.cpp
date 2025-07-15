#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

// CPU grayscale function
void grayscaleCPU(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            uchar gray = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
            output.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }
}

int main() {
    // Load input image
    cv::Mat input = cv::imread("example.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Failed to load image!\n";
        return 1;
    }

    cv::Mat output;

    // Time the CPU grayscale function
    auto t1 = std::chrono::high_resolution_clock::now();
    grayscaleCPU(input, output);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_ms = t2 - t1;
    std::cout << "CPU Time: " << cpu_ms.count() << " ms\n";

    // Write output to file
    cv::imwrite("output_cpu.jpg", output);

    return 0;
}
