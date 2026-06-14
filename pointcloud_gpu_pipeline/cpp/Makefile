NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17
POINT_TARGET ?= pointcloud_cpp_benchmark
PIPELINE_TARGET ?= pointcloud_cpp_full_pipeline_benchmark

.PHONY: all clean run run-full

all: $(POINT_TARGET) $(PIPELINE_TARGET)

$(POINT_TARGET): pointcloud_cpp_benchmark.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

$(PIPELINE_TARGET): pointcloud_cpp_full_pipeline_benchmark.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

run: $(POINT_TARGET)
	./$(POINT_TARGET)

run-full: $(PIPELINE_TARGET)
	./$(PIPELINE_TARGET)

clean:
	rm -f $(POINT_TARGET) $(PIPELINE_TARGET)
