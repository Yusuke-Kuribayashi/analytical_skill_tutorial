#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
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

struct PlaneResult {
    float a = NAN;
    float b = NAN;
    float c = NAN;
    float d = NAN;
    int inliers = 0;
};

struct PipelineCounts {
    int generated = 0;
    int downsampled = 0;
    int remaining = 0;
    int clusters = 0;
};

struct Config {
    int width = 640;
    int height = 480;
    int runs = 11;
    int warmup_runs = 2;
    int ransac_iterations = 256;
    int cluster_iterations = 64;
    int cluster_min_size = 20;
    float fx = 550.0F;
    float fy = 550.0F;
    float cx = -1.0F;
    float cy = -1.0F;
    float min_depth = 0.0F;
    float max_depth = std::numeric_limits<float>::infinity();
    float voxel_size = 0.03F;
    float plane_threshold = 0.015F;
    float min_inlier_ratio = 0.1F;
    float cluster_tolerance = 0.05F;
    unsigned int seed = 0;
};

struct TimedResult {
    double first_ms = 0.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t status = (call);                                                             \
        if (status != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status)); \
        }                                                                                        \
    } while (0)

__host__ __device__ Point6 operator+(const Point6& lhs, const Point6& rhs) {
    return Point6{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.r + rhs.r, lhs.g + rhs.g, lhs.b + rhs.b};
}

struct IsFinitePoint {
    __host__ __device__ bool operator()(const Point6& point) const {
        return isfinite(point.z);
    }
};

struct IsRemainingPoint {
    const std::uint8_t* mask;
    explicit IsRemainingPoint(const std::uint8_t* value) : mask(value) {}
    __host__ __device__ bool operator()(const thrust::tuple<Point6, int>& item) const {
        return mask[thrust::get<1>(item)] != 0;
    }
};

struct NegativeLabel {
    __host__ __device__ bool operator()(int label) const {
        return label < 0;
    }
};

struct MinClusterSize {
    int min_size;
    explicit MinClusterSize(int value) : min_size(value) {}
    __host__ __device__ bool operator()(int size) const {
        return size >= min_size;
    }
};

__host__ __device__ std::int64_t voxel_key(float x, float y, float z, float voxel_size) {
    constexpr std::int64_t offset = 1LL << 20;
    constexpr int bits = 21;
    std::int64_t ix = static_cast<std::int64_t>(floorf(x / voxel_size)) + offset;
    std::int64_t iy = static_cast<std::int64_t>(floorf(y / voxel_size)) + offset;
    std::int64_t iz = static_cast<std::int64_t>(floorf(z / voxel_size)) + offset;
    return (ix << (bits * 2)) | (iy << bits) | iz;
}

__global__ void rgbd_to_dense_kernel(
    const Rgb* rgb,
    const float* depth,
    Point6* output,
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

__global__ void compute_voxel_keys_kernel(const Point6* points, std::int64_t* keys, int count, float voxel_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    Point6 point = points[index];
    keys[index] = voxel_key(point.x, point.y, point.z, voxel_size);
}

__global__ void divide_points_kernel(Point6* points, const int* counts, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float inv = 1.0F / static_cast<float>(counts[index]);
    points[index].x *= inv;
    points[index].y *= inv;
    points[index].z *= inv;
    points[index].r *= inv;
    points[index].g *= inv;
    points[index].b *= inv;
}

__global__ void ransac_count_kernel(
    const Point6* points,
    const int3* samples,
    int* counts,
    float4* coefficients,
    int point_count,
    int iterations,
    float threshold) {
    int iteration = blockIdx.x * blockDim.x + threadIdx.x;
    if (iteration >= iterations) {
        return;
    }
    int3 sample = samples[iteration];
    Point6 p0 = points[sample.x];
    Point6 p1 = points[sample.y];
    Point6 p2 = points[sample.z];
    float ux = p1.x - p0.x;
    float uy = p1.y - p0.y;
    float uz = p1.z - p0.z;
    float vx = p2.x - p0.x;
    float vy = p2.y - p0.y;
    float vz = p2.z - p0.z;
    float nx = uy * vz - uz * vy;
    float ny = uz * vx - ux * vz;
    float nz = ux * vy - uy * vx;
    float norm = sqrtf(nx * nx + ny * ny + nz * nz);
    if (norm <= 0.0F || !isfinite(norm)) {
        counts[iteration] = 0;
        coefficients[iteration] = make_float4(NAN, NAN, NAN, NAN);
        return;
    }
    nx /= norm;
    ny /= norm;
    nz /= norm;
    float d = -(nx * p0.x + ny * p0.y + nz * p0.z);
    int inliers = 0;
    for (int index = 0; index < point_count; ++index) {
        Point6 point = points[index];
        float distance = fabsf(point.x * nx + point.y * ny + point.z * nz + d);
        if (distance <= threshold) {
            ++inliers;
        }
    }
    counts[iteration] = inliers;
    coefficients[iteration] = make_float4(nx, ny, nz, d);
}

__global__ void mark_remaining_kernel(
    const Point6* points,
    std::uint8_t* remaining,
    int point_count,
    float4 coefficient,
    float threshold) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count) {
        return;
    }
    Point6 point = points[index];
    float distance = fabsf(point.x * coefficient.x + point.y * coefficient.y + point.z * coefficient.z + coefficient.w);
    remaining[index] = distance > threshold ? 1 : 0;
}

__global__ void init_labels_kernel(int* labels, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        labels[index] = index;
    }
}

__global__ void propagate_labels_kernel(const Point6* points, int* labels, int* changed, int count, float tolerance_sq) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * count;
    if (pair >= total) {
        return;
    }
    int left = pair / count;
    int right = pair - left * count;
    if (right <= left) {
        return;
    }
    Point6 a = points[left];
    Point6 b = points[right];
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    if (dx * dx + dy * dy + dz * dz > tolerance_sq) {
        return;
    }
    int left_label = labels[left];
    int right_label = labels[right];
    int minimum = left_label < right_label ? left_label : right_label;
    int old_left = atomicMin(labels + left, minimum);
    int old_right = atomicMin(labels + right, minimum);
    if (old_left > minimum || old_right > minimum) {
        *changed = 1;
    }
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--width") config.width = std::stoi(value(arg));
        else if (arg == "--height") config.height = std::stoi(value(arg));
        else if (arg == "--runs") config.runs = std::stoi(value(arg));
        else if (arg == "--warmup-runs") config.warmup_runs = std::stoi(value(arg));
        else if (arg == "--ransac-iterations") config.ransac_iterations = std::stoi(value(arg));
        else if (arg == "--cluster-iterations") config.cluster_iterations = std::stoi(value(arg));
        else if (arg == "--cluster-min-size") config.cluster_min_size = std::stoi(value(arg));
        else if (arg == "--fx") config.fx = std::stof(value(arg));
        else if (arg == "--fy") config.fy = std::stof(value(arg));
        else if (arg == "--cx") config.cx = std::stof(value(arg));
        else if (arg == "--cy") config.cy = std::stof(value(arg));
        else if (arg == "--voxel-size") config.voxel_size = std::stof(value(arg));
        else if (arg == "--plane-threshold") config.plane_threshold = std::stof(value(arg));
        else if (arg == "--cluster-tolerance") config.cluster_tolerance = std::stof(value(arg));
        else if (arg == "--seed") config.seed = static_cast<unsigned int>(std::stoul(value(arg)));
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (config.cx < 0.0F) config.cx = (static_cast<float>(config.width) - 1.0F) * 0.5F;
    if (config.cy < 0.0F) config.cy = (static_cast<float>(config.height) - 1.0F) * 0.5F;
    return config;
}

void generate_rgbd(const Config& config, std::vector<Rgb>& rgb, std::vector<float>& depth) {
    int total = config.width * config.height;
    rgb.resize(total);
    depth.resize(total);
    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; ++x) {
            int index = y * config.width + x;
            rgb[index] = Rgb{static_cast<std::uint8_t>(x % 256), static_cast<std::uint8_t>(y % 256), static_cast<std::uint8_t>((x + y) % 256)};
            float z = 1.0F + 0.00015F * static_cast<float>(x) + 0.00010F * static_cast<float>(y);
            int cx = config.width / 2;
            int cy = config.height / 2;
            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy < (config.height / 7) * (config.height / 7)) {
                z -= 0.22F;
            }
            if ((x * 13 + y * 7) % 97 == 0) {
                z = std::numeric_limits<float>::quiet_NaN();
            }
            depth[index] = z;
        }
    }
}

std::vector<int3> make_samples(int count, int iterations, unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<int3> samples(iterations);
    for (int i = 0; i < iterations; ++i) {
        std::uniform_int_distribution<int> dist(0, count - 1);
        int a = dist(rng);
        int b = dist(rng);
        int c = dist(rng);
        while (b == a) b = dist(rng);
        while (c == a || c == b) c = dist(rng);
        samples[i] = make_int3(a, b, c);
    }
    return samples;
}

std::vector<Point6> generate_points_cpu(const std::vector<Rgb>& rgb, const std::vector<float>& depth, const Config& config) {
    std::vector<Point6> points;
    points.reserve(depth.size());
    for (int index = 0; index < static_cast<int>(depth.size()); ++index) {
        int y = index / config.width;
        int x = index - y * config.width;
        float z = depth[index];
        if (!std::isfinite(z) || z <= config.min_depth || z >= config.max_depth) continue;
        points.push_back(Point6{(static_cast<float>(x) - config.cx) * z / config.fx, (static_cast<float>(y) - config.cy) * z / config.fy, z, static_cast<float>(rgb[index].r), static_cast<float>(rgb[index].g), static_cast<float>(rgb[index].b)});
    }
    return points;
}

std::vector<Point6> voxel_cpu(const std::vector<Point6>& points, float voxel_size) {
    std::vector<std::pair<std::int64_t, Point6>> keyed;
    keyed.reserve(points.size());
    for (const auto& point : points) {
        keyed.emplace_back(voxel_key(point.x, point.y, point.z, voxel_size), point);
    }
    std::sort(keyed.begin(), keyed.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    std::vector<Point6> output;
    output.reserve(keyed.size());
    for (std::size_t index = 0; index < keyed.size();) {
        std::int64_t key = keyed[index].first;
        Point6 sum{};
        int count = 0;
        while (index < keyed.size() && keyed[index].first == key) {
            sum = sum + keyed[index].second;
            ++count;
            ++index;
        }
        float inv = 1.0F / static_cast<float>(count);
        output.push_back(Point6{sum.x * inv, sum.y * inv, sum.z * inv, sum.r * inv, sum.g * inv, sum.b * inv});
    }
    return output;
}

PlaneResult ransac_cpu(const std::vector<Point6>& points, const Config& config) {
    auto samples = make_samples(static_cast<int>(points.size()), config.ransac_iterations, config.seed);
    PlaneResult best;
    for (const auto& sample : samples) {
        Point6 p0 = points[sample.x], p1 = points[sample.y], p2 = points[sample.z];
        float ux = p1.x - p0.x, uy = p1.y - p0.y, uz = p1.z - p0.z;
        float vx = p2.x - p0.x, vy = p2.y - p0.y, vz = p2.z - p0.z;
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (norm <= 0.0F || !std::isfinite(norm)) continue;
        nx /= norm; ny /= norm; nz /= norm;
        float d = -(nx * p0.x + ny * p0.y + nz * p0.z);
        int inliers = 0;
        for (const auto& point : points) {
            if (std::fabs(point.x * nx + point.y * ny + point.z * nz + d) <= config.plane_threshold) ++inliers;
        }
        if (inliers > best.inliers) best = PlaneResult{nx, ny, nz, d, inliers};
    }
    return best;
}

std::vector<Point6> remove_plane_cpu(const std::vector<Point6>& points, const PlaneResult& plane, float threshold) {
    std::vector<Point6> remaining;
    remaining.reserve(points.size());
    for (const auto& point : points) {
        float distance = std::fabs(point.x * plane.a + point.y * plane.b + point.z * plane.c + plane.d);
        if (distance > threshold) remaining.push_back(point);
    }
    return remaining;
}

int cluster_cpu(const std::vector<Point6>& points, const Config& config) {
    if (points.empty()) return 0;
    float tolerance_sq = config.cluster_tolerance * config.cluster_tolerance;
    std::vector<int> labels(points.size(), -1);
    int clusters = 0;
    for (int start = 0; start < static_cast<int>(points.size()); ++start) {
        if (labels[start] >= 0) continue;
        std::deque<int> queue;
        std::vector<int> members;
        labels[start] = clusters;
        queue.push_back(start);
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop_front();
            members.push_back(current);
            for (int other = 0; other < static_cast<int>(points.size()); ++other) {
                if (labels[other] >= 0) continue;
                float dx = points[current].x - points[other].x;
                float dy = points[current].y - points[other].y;
                float dz = points[current].z - points[other].z;
                if (dx * dx + dy * dy + dz * dz <= tolerance_sq) {
                    labels[other] = clusters;
                    queue.push_back(other);
                }
            }
        }
        if (static_cast<int>(members.size()) >= config.cluster_min_size) ++clusters;
        else for (int index : members) labels[index] = -2;
    }
    return clusters;
}

PipelineCounts run_cpu_pipeline(const std::vector<Rgb>& rgb, const std::vector<float>& depth, const Config& config) {
    auto points = generate_points_cpu(rgb, depth, config);
    auto downsampled = voxel_cpu(points, config.voxel_size);
    auto plane = ransac_cpu(downsampled, config);
    auto remaining = remove_plane_cpu(downsampled, plane, config.plane_threshold);
    int clusters = cluster_cpu(remaining, config);
    return PipelineCounts{static_cast<int>(points.size()), static_cast<int>(downsampled.size()), static_cast<int>(remaining.size()), clusters};
}

thrust::device_vector<Point6> generate_points_gpu(const thrust::device_vector<Rgb>& rgb, const thrust::device_vector<float>& depth, const Config& config) {
    int total = config.width * config.height;
    thrust::device_vector<Point6> dense(total);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rgbd_to_dense_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(rgb.data()), thrust::raw_pointer_cast(depth.data()), thrust::raw_pointer_cast(dense.data()), config.width, config.height, config.fx, config.fy, config.cx, config.cy, config.min_depth, config.max_depth);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_vector<Point6> points(total);
    auto end = thrust::copy_if(dense.begin(), dense.end(), points.begin(), IsFinitePoint{});
    points.resize(static_cast<std::size_t>(end - points.begin()));
    return points;
}

thrust::device_vector<Point6> voxel_gpu(thrust::device_vector<Point6>& points, const Config& config) {
    int count = static_cast<int>(points.size());
    thrust::device_vector<std::int64_t> keys(count);
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    compute_voxel_keys_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(keys.data()), count, config.voxel_size);
    CUDA_CHECK(cudaGetLastError());
    thrust::sort_by_key(keys.begin(), keys.end(), points.begin());
    thrust::device_vector<std::int64_t> unique_keys(count);
    thrust::device_vector<Point6> sums(count);
    auto reduced_points = thrust::reduce_by_key(keys.begin(), keys.end(), points.begin(), unique_keys.begin(), sums.begin());
    int voxel_count = static_cast<int>(reduced_points.first - unique_keys.begin());
    thrust::device_vector<int> ones(count, 1);
    thrust::device_vector<int> counts(count);
    auto reduced_counts = thrust::reduce_by_key(keys.begin(), keys.end(), ones.begin(), unique_keys.begin(), counts.begin());
    (void)reduced_counts;
    sums.resize(voxel_count);
    counts.resize(voxel_count);
    blocks = (voxel_count + threads - 1) / threads;
    divide_points_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(sums.data()), thrust::raw_pointer_cast(counts.data()), voxel_count);
    CUDA_CHECK(cudaGetLastError());
    return sums;
}

PlaneResult ransac_gpu(thrust::device_vector<Point6>& points, const Config& config) {
    int count = static_cast<int>(points.size());
    auto host_samples = make_samples(count, config.ransac_iterations, config.seed);
    thrust::device_vector<int3> samples = host_samples;
    thrust::device_vector<int> counts(config.ransac_iterations);
    thrust::device_vector<float4> coefficients(config.ransac_iterations);
    int threads = 128;
    int blocks = (config.ransac_iterations + threads - 1) / threads;
    ransac_count_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(samples.data()), thrust::raw_pointer_cast(counts.data()), thrust::raw_pointer_cast(coefficients.data()), count, config.ransac_iterations, config.plane_threshold);
    CUDA_CHECK(cudaGetLastError());
    auto best_iter = thrust::max_element(counts.begin(), counts.end());
    int best_index = static_cast<int>(best_iter - counts.begin());
    int best_count = counts[best_index];
    float4 coeff = coefficients[best_index];
    return PlaneResult{coeff.x, coeff.y, coeff.z, coeff.w, best_count};
}

thrust::device_vector<Point6> remove_plane_gpu(thrust::device_vector<Point6>& points, const PlaneResult& plane, const Config& config) {
    int count = static_cast<int>(points.size());
    thrust::device_vector<std::uint8_t> mask(count);
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    float4 coeff = make_float4(plane.a, plane.b, plane.c, plane.d);
    mark_remaining_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(mask.data()), count, coeff, config.plane_threshold);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_vector<Point6> remaining(count);
    auto index_begin = thrust::make_counting_iterator(0);
    auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(points.begin(), index_begin));
    auto zipped_end = thrust::make_zip_iterator(thrust::make_tuple(points.end(), index_begin + count));
    auto end = thrust::copy_if(points.begin(), points.end(), zipped_begin, remaining.begin(), IsRemainingPoint(thrust::raw_pointer_cast(mask.data())));
    remaining.resize(static_cast<std::size_t>(end - remaining.begin()));
    return remaining;
}

int cluster_gpu(thrust::device_vector<Point6>& points, const Config& config) {
    int count = static_cast<int>(points.size());
    if (count == 0) return 0;
    thrust::device_vector<int> labels(count);
    thrust::device_vector<int> changed(1);
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    init_labels_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(labels.data()), count);
    CUDA_CHECK(cudaGetLastError());
    int pair_count = count * count;
    int pair_blocks = (pair_count + threads - 1) / threads;
    for (int i = 0; i < config.cluster_iterations; ++i) {
        thrust::fill(changed.begin(), changed.end(), 0);
        propagate_labels_kernel<<<pair_blocks, threads>>>(thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(labels.data()), thrust::raw_pointer_cast(changed.data()), count, config.cluster_tolerance * config.cluster_tolerance);
        CUDA_CHECK(cudaGetLastError());
        if (changed[0] == 0) break;
    }
    thrust::sort(labels.begin(), labels.end());
    auto valid_begin = thrust::remove_if(labels.begin(), labels.end(), NegativeLabel{});
    labels.erase(valid_begin, labels.end());
    if (labels.empty()) return 0;
    thrust::device_vector<int> unique_labels(labels.size());
    thrust::device_vector<int> label_counts(labels.size());
    auto reduced = thrust::reduce_by_key(labels.begin(), labels.end(), thrust::make_constant_iterator(1), unique_labels.begin(), label_counts.begin());
    int group_count = static_cast<int>(reduced.first - unique_labels.begin());
    label_counts.resize(group_count);
    return static_cast<int>(thrust::count_if(label_counts.begin(), label_counts.end(), MinClusterSize(config.cluster_min_size)));
}

PipelineCounts run_gpu_pipeline(const std::vector<Rgb>& host_rgb, const std::vector<float>& host_depth, const Config& config) {
    thrust::device_vector<Rgb> rgb = host_rgb;
    thrust::device_vector<float> depth = host_depth;
    auto points = generate_points_gpu(rgb, depth, config);
    int generated = static_cast<int>(points.size());
    auto downsampled = voxel_gpu(points, config);
    int downsampled_count = static_cast<int>(downsampled.size());
    auto plane = ransac_gpu(downsampled, config);
    auto remaining = remove_plane_gpu(downsampled, plane, config);
    int remaining_count = static_cast<int>(remaining.size());
    int clusters = cluster_gpu(remaining, config);
    CUDA_CHECK(cudaDeviceSynchronize());
    return PipelineCounts{generated, downsampled_count, remaining_count, clusters};
}

TimedResult summarize(const std::vector<double>& timings, int warmup) {
    TimedResult result;
    result.first_ms = timings.front();
    auto first = timings.begin() + warmup;
    auto last = timings.end();
    result.mean_ms = std::accumulate(first, last, 0.0) / static_cast<double>(last - first);
    result.min_ms = *std::min_element(first, last);
    result.max_ms = *std::max_element(first, last);
    return result;
}

void print_result(const std::string& name, const TimedResult& time, const PipelineCounts& counts) {
    std::cout << std::left << std::setw(20) << name
              << " first_ms=" << std::fixed << std::setprecision(3) << time.first_ms
              << " mean_ms=" << time.mean_ms
              << " min_ms=" << time.min_ms
              << " max_ms=" << time.max_ms
              << " generated=" << counts.generated
              << " downsampled=" << counts.downsampled
              << " remaining=" << counts.remaining
              << " clusters=" << counts.clusters << '\n';
}

int main(int argc, char** argv) {
    try {
        Config config = parse_args(argc, argv);
        std::vector<Rgb> rgb;
        std::vector<float> depth;
        generate_rgbd(config, rgb, depth);
        std::vector<double> cpu_times;
        std::vector<double> gpu_times;
        PipelineCounts cpu_counts;
        PipelineCounts gpu_counts;
        cpu_times.reserve(config.runs);
        gpu_times.reserve(config.runs);
        for (int run = 0; run < config.runs; ++run) {
            auto start = std::chrono::steady_clock::now();
            cpu_counts = run_cpu_pipeline(rgb, depth, config);
            auto end = std::chrono::steady_clock::now();
            cpu_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        for (int run = 0; run < config.runs; ++run) {
            auto start = std::chrono::steady_clock::now();
            gpu_counts = run_gpu_pipeline(rgb, depth, config);
            auto end = std::chrono::steady_clock::now();
            gpu_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        int device = 0;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "device=" << prop.name << " image=" << config.width << "x" << config.height
                  << " runs=" << config.runs << " warmup=" << config.warmup_runs
                  << " voxel=" << config.voxel_size << " ransac=" << config.ransac_iterations
                  << " cluster_iter=" << config.cluster_iterations << '\n';
        print_result("cpp_cpu_pipeline", summarize(cpu_times, config.warmup_runs), cpu_counts);
        print_result("cuda_pipeline", summarize(gpu_times, config.warmup_runs), gpu_counts);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
