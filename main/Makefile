# === CONFIGURATION ===
TARGET = image_filter
SRC_CPP = main.cpp
SRC_CU  = filter.cu

# Use pkg-config to get OpenCV flags
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# CUDA compiler
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Xcompiler -fPIC

# === BUILD RULES ===

all: $(TARGET)

$(TARGET): $(SRC_CPP) $(SRC_CU)
	$(NVCC) $(SRC_CPP) $(SRC_CU) -o $@ $(CXXFLAGS) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)

