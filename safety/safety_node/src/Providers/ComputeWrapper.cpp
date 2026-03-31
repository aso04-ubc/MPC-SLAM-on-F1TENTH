#include "ComputeWrapper.hpp"
#include <cstring>

#ifdef APP_USE_CUDA
#include "DeviceComputation.cuh"

class CudaDepthMapToBucketComputer : public IDepthMapToBucketComputer {
public:
    void Compute(const DepthMapToBucketComputeInfo &info) override;

    CudaDepthMapToBucketComputer() = default;

private:
    template<typename T>
    struct CudaDeallocator {
        void operator()(T *ptr) const {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    template<typename T>
    struct DeviceVisibleBuffer {
        DeviceVisibleBuffer() = default;

        std::unique_ptr<T, CudaDeallocator<T>> Buffer;
        uint32_t Size;

        DeviceVisibleBuffer(uint32_t element_count) : Size(element_count) {
            T *rawPtr = nullptr;
            cudaError_t err = cudaMallocManaged(&rawPtr, element_count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate device buffer: " + std::string(cudaGetErrorString(err)));
            }
            Buffer.reset(rawPtr);
        }
    };

private:
    DeviceVisibleBuffer<uint16_t> DepthMapBuffer;
    DeviceVisibleBuffer<uint32_t> BucketBuffer;
};

void CudaDepthMapToBucketComputer::Compute(const DepthMapToBucketComputeInfo &info) {
    ComputeDepthBucketKernelSubmissionInfo submissionInfo;
    size_t requiredSize = info.DepthMapInMillimeter.Width * info.DepthMapInMillimeter.Height;
    if (DepthMapBuffer.Size < requiredSize) {
        DepthMapBuffer = DeviceVisibleBuffer<uint16_t>(requiredSize);
    }
    std::memcpy(DepthMapBuffer.Buffer.get(), info.DepthMapInMillimeter.Buffer, requiredSize * sizeof(uint16_t));

    size_t bucketCount = info.Buckets.Count;
    if (BucketBuffer.Size < bucketCount) {
        BucketBuffer = DeviceVisibleBuffer<uint32_t>(bucketCount);
    }
    std::memset(BucketBuffer.Buffer.get(), 0, bucketCount * sizeof(uint32_t));

    submissionInfo.Configuration.BucketStepSizeInMillimeter = info.Buckets.StepSizeInMillimeter;
    submissionInfo.Configuration.RegionOfInterestStartX = info.RegionOfInterest.StartX;
    submissionInfo.Configuration.RegionOfInterestEndX = info.RegionOfInterest.EndX;
    submissionInfo.Configuration.RegionOfInterestStartY = info.RegionOfInterest.StartY;
    submissionInfo.Configuration.RegionOfInterestEndY = info.RegionOfInterest.EndY;

    submissionInfo.DeviceVisible.Read.DepthMapInMillimeterBuffer = DepthMapBuffer.Buffer.get();
    submissionInfo.DeviceVisible.Read.DepthMapWidth = info.DepthMapInMillimeter.Width;
    submissionInfo.DeviceVisible.Read.DepthMapHeight = info.DepthMapInMillimeter.Height;

    submissionInfo.DeviceVisible.Write.BucketBuffer = BucketBuffer.Buffer.get();
    submissionInfo.DeviceVisible.Write.BucketCount = bucketCount;

    uint32_t workWidth = submissionInfo.Configuration.RegionOfInterestEndX - submissionInfo.Configuration.
                         RegionOfInterestStartX;
    uint32_t workHeight = submissionInfo.Configuration.RegionOfInterestEndY - submissionInfo.Configuration.
                          RegionOfInterestStartY;

    if (workWidth == 0 || workHeight == 0) {
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (workWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (workHeight + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // printf("[CUDA Host] Launching kernel with grid(%u,%u) blocks(%u,%u), ROI: X[%u,%u) Y[%u,%u)\n",
    //        blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y,
    //        submissionInfo.Configuration.RegionOfInterestStartX,
    //        submissionInfo.Configuration.RegionOfInterestEndX,
    //        submissionInfo.Configuration.RegionOfInterestStartY,
    //        submissionInfo.Configuration.RegionOfInterestEndY);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Pre-kernel sync failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA pre-kernel sync failed: " + std::string(cudaGetErrorString(err)));
    }

    // Launch kernel
    ComputeDepthBucketKernelLaunch(submissionInfo, blocksPerGrid, threadsPerBlock);

    // Check for kernel launch errors
    err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    //     throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    // }

    // Wait for kernel to complete and check for execution errors
    err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "[CUDA ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
    //     throw std::runtime_error("CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
    // }

    // printf("[CUDA Host] Kernel completed successfully\n");

    // Copy results back
    std::memcpy(info.Buckets.Buffer, BucketBuffer.Buffer.get(), bucketCount * sizeof(uint32_t));

    // Debug: Print first few buckets
    // printf("[CUDA Host] All buckets computed. First few bucket counts: ");
    // for (size_t i = 0; i < std::min(400UL, bucketCount); ++i) {
    //     printf("%u ", info.Buckets.Buffer[i]);
    // }
    // printf("\n");
}

#else

#endif

class CPUDepthMapToBucketComputer : public IDepthMapToBucketComputer {
public:
    void Compute(const DepthMapToBucketComputeInfo &info) override {
        const auto &depthMap = info.DepthMapInMillimeter;
        const auto &buckets = info.Buckets;
        const auto &roi = info.RegionOfInterest;

        // Clear buckets
        std::memset(buckets.Buffer, 0, buckets.Count * sizeof(uint32_t));

        // ROI is treated as half-open intervals: x in [StartX, EndX) and y in [StartY, EndY).
        // Depth values are assumed to be in millimeters (provided by the depth map message).
        for (uint32_t y = roi.StartY; y < roi.EndY; ++y) {
            for (uint32_t x = roi.StartX; x < roi.EndX; ++x) {
                uint32_t index = y * depthMap.Width + x;
                uint16_t depthValue = depthMap.Buffer[index];

                if (depthValue == 0) {
                    continue; // Skip invalid depth
                }

                // Integer division maps depthValue to a bucket:
                // bucket i roughly corresponds to distances in [i*step, (i+1)*step).
                uint32_t bucketIndex = depthValue / buckets.StepSizeInMillimeter;
                if (bucketIndex < buckets.Count) {
                    buckets.Buffer[bucketIndex]++;
                }
            }
        }
    }
};

std::unique_ptr<IDepthMapToBucketComputer> IDepthMapToBucketComputer::Create(ComputeMode mode) {
    switch (mode == ComputeMode::eAuto ? AutoDetectComputeMode() : mode) {
        case ComputeMode::eHost:
            return std::make_unique<CPUDepthMapToBucketComputer>();
        case ComputeMode::eDeviceAccelerated:
#ifdef APP_USE_CUDA
            return std::make_unique<CudaDepthMapToBucketComputer>();
#else
            throw std::runtime_error("Device accelerated computation is not supported in this build.");
#endif
        default:
            throw std::runtime_error("Unknown ComputeMode specified.");
    }
}

std::optional<uint32_t> GetLowestProportionFromDepthMap(IDepthMapToBucketComputer &computer,
                                                        std::vector<uint32_t> &scratchBuckets,
                                                        const LowestProportionFromDepthMapComputeInfo &info) {
    const auto &config = info.Config;
    const uint32_t step = config.StepSizeInMillimeter;

    // 1. Histogram range: We need buckets up to DistanceMax
    size_t bucketCount = (config.DistanceMaxInMillimeter / step) + 1;
    // std::vector<uint32_t> buckets(bucketCount, 0);
    if (scratchBuckets.size() < bucketCount) {
        scratchBuckets.resize(bucketCount, 0);
    } else {
        std::fill(scratchBuckets.begin(), scratchBuckets.end(), 0);
    }

    BucketInfo bucketInfo;
    bucketInfo.Buffer = scratchBuckets.data();
    bucketInfo.Count = static_cast<uint32_t>(bucketCount);
    bucketInfo.StepSizeInMillimeter = step;

    DepthMapToBucketComputeInfo computeInfo;
    computeInfo.DepthMapInMillimeter = info.DepthMapInMillimeter;
    computeInfo.Buckets = bucketInfo;
    computeInfo.Buckets.StepSizeInMillimeter = step;
    computeInfo.RegionOfInterest = info.RegionOfInterest;

    // Execute GPU Kernel
    computer.Compute(computeInfo);

    // 2. Define the focus window in the histogram
    // startBucket: the first bucket that contains DistanceMin
    size_t startBucket = config.DistanceMinInMillimeter / step;
    // endBucket: the last bucket that contains DistanceMax
    size_t endBucket = config.DistanceMaxInMillimeter / step;

    // Safety check: ensure we don't exceed vector bounds
    endBucket = std::min(endBucket, bucketCount - 1);

    // 3. Sum up ONLY pixels within [DistanceMin, DistanceMax]
    // (inclusive bucket range: i in [startBucket, endBucket]).
    uint32_t totalCount = 0;
    for (size_t i = startBucket; i <= endBucket; ++i) {
        totalCount += scratchBuckets[i];
    }

    if (totalCount == 0) {
        return std::nullopt; // Let caller handle no valid data case
    }

    // 4. Find the distance where the cumulative proportion is met
    auto targetThreshold = static_cast<uint32_t>(totalCount * config.Proportion);
    uint32_t cumulativeCount = 0;

    for (size_t i = startBucket; i <= endBucket; ++i) {
        cumulativeCount += scratchBuckets[i];
        if (cumulativeCount >= targetThreshold) {
            // Map bucket index back to a representative distance:
            // we use the bucket's lower edge (i * step) as the returned millimeter value.
            return static_cast<uint32_t>(i * step);
        }
    }

    return config.DistanceMaxInMillimeter;
}
