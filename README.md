# Description
A GPU-accelerated image processing application that applies color filters such as grayscale and sepia to high-resolution images using CUDA and OpenCV.

## Features
- High-performance CUDA kernels for per-pixel parallel processing
- Supports multiple filters with real-time performance measurement
- Flexible design allowing easy addition of new filters
- Command-line interface to select desired filter dynamically

## Requirements
- CUDA-enabled GPU
- OpenCV 
- CUDA Toolkit

## Build and Run
- ```cd``` into the main directory
- run ```make```
- Add jpg into the directory (default name should be example.jpg)
- run ```./image_filter```
