# === File names ===
CPU_SRC = benchmarkCPU.cpp

# === Output binary names ===
CPU_BIN = CPUbench

# === Compiler settings ===
CXX = g++

# === OpenCV flags (change opencv4 to opencv if needed) ===
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# === Build rules ===
all: $(CPU_BIN) $(CUDA_BIN)

$(CPU_BIN): $(CPU_SRC)
	$(CXX) $(CPU_SRC) -o $(CPU_BIN) $(OPENCV_FLAGS)

clean:
	rm -f $(CPU_BIN) 