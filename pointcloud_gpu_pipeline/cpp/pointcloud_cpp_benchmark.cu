#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Rgb {
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
};

struct Point6 {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
};

struct Config {
    int width = 1280;
    int height = 720;
    int runs = 31;
    int warmup_runs = 5;
    float fx = 1100.0F;
    float fy = 1100.0F;
    float cx = -1.0F;
    float cy = -1.0F;
    float min_depth = 0.0F;
    float max_depth = std::numeric_limits<float>::infinity();
};

struct TimedResult {
    double first_ms = 0.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t status = (call);                                                              \
        if (status != cudaSuccess) {                                                              \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));  \
        }                                                                                         \
    } while (0)

__global__ void rgbd_to_dense_pointcloud_kernel(
    const Rgb* rgb,
    const float* depth,
    Point6* output,
    int* valid_count,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    float min_depth,
    float max_depth) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int point_count = width * height;
    if (index >= point_count) {
        return;
    }

    int y = index / width;
    int x = index - y * width;
    float z = depth[index];
    bool valid = isfinite(z) && min_depth < z && z < max_depth;
    Point6 point{};
    if (valid) {
        point.x = (static_cast<float>(x) - cx) * z / fx;
        point.y = (static_cast<float>(y) - cy) * z / fy;
        point.z = z;
        atomicAdd(valid_count, 1);
    } else {
        point.x = NAN;
        point.y = NAN;
        point.z = NAN;
    }
    point.r = static_cast<float>(rgb[index].r);
    point.g = static_cast<float>(rgb[index].g);
    point.b = static_cast<float>(rgb[index].b);
    output[index] = point;
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--width") {
            config.width = std::stoi(require_value(arg));
        } else if (arg == "--height") {
            config.height = std::stoi(require_value(arg));
        } else if (arg == "--runs") {
            config.runs = std::stoi(require_value(arg));
        } else if (arg == "--warmup-runs") {
            config.warmup_runs = std::stoi(require_value(arg));
        } else if (arg == "--fx") {
            config.fx = std::stof(require_value(arg));
        } else if (arg == "--fy") {
            config.fy = std::stof(require_value(arg));
        } else if (arg == "--cx") {
            config.cx = std::stof(require_value(arg));
        } else if (arg == "--cy") {
            config.cy = std::stof(require_value(arg));
        } else if (arg == "--min-depth") {
            config.min_depth = std::stof(require_value(arg));
        } else if (arg == "--max-depth") {
            config.max_depth = std::stof(require_value(arg));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (config.width <= 0 || config.height <= 0) {
        throw std::invalid_argument("width and height must be positive");
    }
    if (config.runs <= 0 || config.warmup_runs < 0 || config.warmup_runs >= config.runs) {
        throw std::invalid_argument("runs must be positive and warmup-runs must be less than runs");
    }
    if (config.fx <= 0.0F || config.fy <= 0.0F) {
        throw std::invalid_argument("fx and fy must be positive");
    }
    if (config.cx < 0.0F) {
        config.cx = (static_cast<float>(config.width) - 1.0F) * 0.5F;
    }
    if (config.cy < 0.0F) {
        config.cy = (static_cast<float>(config.height) - 1.0F) * 0.5F;
    }
    return config;
}

void generate_rgbd(const Config& config, std::vector<Rgb>& rgb, std::vector<float>& depth) {
    int point_count = config.width * config.height;
    rgb.resize(point_count);
    depth.resize(point_count);
    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; ++x) {
            int index = y * config.width + x;
            rgb[index] = Rgb{
                static_cast<std::uint8_t>(x % 256),
                static_cast<std::uint8_t>(y % 256),
                static_cast<std::uint8_t>((x + y) % 256),
            };
            float wave = 0.03F * std::sin(static_cast<float>(x) * 0.017F)
                + 0.02F * std::cos(static_cast<float>(y) * 0.011F);
            float object = 0.0F;
            int dx = x - config.width / 2;
            int dy = y - config.height / 2;
            if (dx * dx + dy * dy < (config.height / 6) * (config.height / 6)) {
                object = -0.25F;
            }
            depth[index] = 1.2F + wave + object;
            if ((x * 13 + y * 7) % 97 == 0) {
                depth[index] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

int create_pointcloud_cpu(
    const std::vector<Rgb>& rgb,
    const std::vector<float>& depth,
    std::vector<Point6>& output,
    const Config& config) {
    int valid_count = 0;
    int point_count = config.width * config.height;
    output.resize(point_count);
    for (int index = 0; index < point_count; ++index) {
        int y = index / config.width;
        int x = index - y * config.width;
        float z = depth[index];
        bool valid = std::isfinite(z) && config.min_depth < z && z < config.max_depth;
        Point6 point{};
        if (valid) {
            point.x = (static_cast<float>(x) - config.cx) * z / config.fx;
            point.y = (static_cast<float>(y) - config.cy) * z / config.fy;
            point.z = z;
            ++valid_count;
        } else {
            point.x = std::numeric_limits<float>::quiet_NaN();
            point.y = std::numeric_limits<float>::quiet_NaN();
            point.z = std::numeric_limits<float>::quiet_NaN();
        }
        point.r = static_cast<float>(rgb[index].r);
        point.g = static_cast<float>(rgb[index].g);
        point.b = static_cast<float>(rgb[index].b);
        output[index] = point;
    }
    return valid_count;
}

TimedResult summarize(const std::vector<double>& timings, int warmup_runs) {
    TimedResult result;
    result.first_ms = timings.front();
    auto first = timings.begin() + warmup_runs;
    auto last = timings.end();
    result.mean_ms = std::accumulate(first, last, 0.0) / static_cast<double>(last - first);
    result.min_ms = *std::min_element(first, last);
    result.max_ms = *std::max_element(first, last);
    return result;
}

std::vector<double> benchmark_cpu(
    const std::vector<Rgb>& rgb,
    const std::vector<float>& depth,
    std::vector<Point6>& output,
    const Config& config,
    int& valid_count) {
    std::vector<double> timings;
    timings.reserve(config.runs);
    for (int run = 0; run < config.runs; ++run) {
        auto start = std::chrono::steady_clock::now();
        valid_count = create_pointcloud_cpu(rgb, depth, output, config);
        auto end = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return timings;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

std::vector<double> benchmark_gpu_resident(
    const Rgb* rgb_device,
    const float* depth_device,
    Point6* output_device,
    int* valid_count_device,
    const Config& config,
    int& valid_count) {
    int point_count = config.width * config.height;
    int threads = 256;
    int blocks = (point_count + threads - 1) / threads;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> timings;
    timings.reserve(config.runs);
    for (int run = 0; run < config.runs; ++run) {
        CUDA_CHECK(cudaMemset(valid_count_device, 0, sizeof(int)));
        CUDA_CHECK(cudaEventRecord(start));
        rgbd_to_dense_pointcloud_kernel<<<blocks, threads>>>(
            rgb_device,
            depth_device,
            output_device,
            valid_count_device,
            config.width,
            config.height,
            config.fx,
            config.fy,
            config.cx,
            config.cy,
            config.min_depth,
            config.max_depth);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        timings.push_back(static_cast<double>(elapsed_ms(start, stop)));
    }
    CUDA_CHECK(cudaMemcpy(&valid_count, valid_count_device, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return timings;
}

std::vector<double> benchmark_gpu_with_transfer(
    const std::vector<Rgb>& rgb,
    const std::vector<float>& depth,
    Rgb* rgb_device,
    float* depth_device,
    Point6* output_device,
    int* valid_count_device,
    const Config& config,
    int& valid_count) {
    int point_count = config.width * config.height;
    int threads = 256;
    int blocks = (point_count + threads - 1) / threads;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> timings;
    timings.reserve(config.runs);
    for (int run = 0; run < config.runs; ++run) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(rgb_device, rgb.data(), rgb.size() * sizeof(Rgb), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(depth_device, depth.data(), depth.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(valid_count_device, 0, sizeof(int)));
        rgbd_to_dense_pointcloud_kernel<<<blocks, threads>>>(
            rgb_device,
            depth_device,
            output_device,
            valid_count_device,
            config.width,
            config.height,
            config.fx,
            config.fy,
            config.cx,
            config.cy,
            config.min_depth,
            config.max_depth);
        CUDA_CHECK(cudaMemcpy(&valid_count, valid_count_device, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        timings.push_back(static_cast<double>(elapsed_ms(start, stop)));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return timings;
}

void print_result(const std::string& name, const TimedResult& result, int valid_count) {
    std::cout << std::left << std::setw(28) << name
              << " first_ms=" << std::fixed << std::setprecision(3) << result.first_ms
              << " mean_ms=" << result.mean_ms
              << " min_ms=" << result.min_ms
              << " max_ms=" << result.max_ms
              << " valid_points=" << valid_count << '\n';
}

int main(int argc, char** argv) {
    try {
        Config config = parse_args(argc, argv);
        std::vector<Rgb> rgb;
        std::vector<float> depth;
        generate_rgbd(config, rgb, depth);
        std::vector<Point6> cpu_output;
        int point_count = config.width * config.height;
        int cpu_valid_count = 0;
        int gpu_resident_valid_count = 0;
        int gpu_transfer_valid_count = 0;

        Rgb* rgb_device = nullptr;
        float* depth_device = nullptr;
        Point6* output_device = nullptr;
        int* valid_count_device = nullptr;
        CUDA_CHECK(cudaMalloc(&rgb_device, rgb.size() * sizeof(Rgb)));
        CUDA_CHECK(cudaMalloc(&depth_device, depth.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&output_device, static_cast<std::size_t>(point_count) * sizeof(Point6)));
        CUDA_CHECK(cudaMalloc(&valid_count_device, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(rgb_device, rgb.data(), rgb.size() * sizeof(Rgb), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(depth_device, depth.data(), depth.size() * sizeof(float), cudaMemcpyHostToDevice));

        auto cpu_timings = benchmark_cpu(rgb, depth, cpu_output, config, cpu_valid_count);
        auto gpu_resident_timings = benchmark_gpu_resident(
            rgb_device,
            depth_device,
            output_device,
            valid_count_device,
            config,
            gpu_resident_valid_count);
        auto gpu_transfer_timings = benchmark_gpu_with_transfer(
            rgb,
            depth,
            rgb_device,
            depth_device,
            output_device,
            valid_count_device,
            config,
            gpu_transfer_valid_count);

        int device = 0;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "device=" << prop.name << " image=" << config.width << "x" << config.height
                  << " runs=" << config.runs << " warmup=" << config.warmup_runs << '\n';
        print_result("cpp_cpu_dense", summarize(cpu_timings, config.warmup_runs), cpu_valid_count);
        print_result("cuda_kernel_resident", summarize(gpu_resident_timings, config.warmup_runs), gpu_resident_valid_count);
        print_result("cuda_h2d_kernel_count", summarize(gpu_transfer_timings, config.warmup_runs), gpu_transfer_valid_count);

        CUDA_CHECK(cudaFree(rgb_device));
        CUDA_CHECK(cudaFree(depth_device));
        CUDA_CHECK(cudaFree(output_device));
        CUDA_CHECK(cudaFree(valid_count_device));
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
