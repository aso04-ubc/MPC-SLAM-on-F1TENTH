#pragma once

#include <cstdint>
#include <memory>


struct DepthMapInfo {
    // Pointer to depth values stored in millimeters.
    // Expected layout: Buffer has at least Width * Height elements.
    const uint16_t *Buffer;
    uint32_t Width;
    uint32_t Height;
};

struct BucketInfo {
    // Histogram buckets.
    // Buffer size must be at least `Count` and each bucket accumulates pixel counts.
    uint32_t *Buffer;
    uint32_t Count;
    // Bucket width in millimeters (e.g., 10 means bucket i covers [i*10, (i+1)*10)).
    uint32_t StepSizeInMillimeter;
};

struct RegionOfInterestInfo {
    // ROI boundaries in pixel coordinates.
    // The compute implementations treat them as a half-open interval: [StartX, EndX) x [StartY, EndY).
    uint32_t StartX;
    uint32_t EndX;
    uint32_t StartY;
    uint32_t EndY;
};


struct DepthMapToBucketComputeInfo {
    DepthMapInfo DepthMapInMillimeter;

    BucketInfo Buckets;

    RegionOfInterestInfo RegionOfInterest;
};

struct LowestProportionFromDepthMapComputeInfo {
    DepthMapInfo DepthMapInMillimeter;
    RegionOfInterestInfo RegionOfInterest;

    // Compute settings:
    // - Proportion: fraction of valid depth pixels to cover
    // - DistanceMinInMillimeter / DistanceMaxInMillimeter: histogram range
    // - StepSizeInMillimeter: bucket width in millimeters
    struct ConfigType {
        float Proportion;
        uint32_t DistanceMinInMillimeter;
        uint32_t DistanceMaxInMillimeter;
        uint32_t StepSizeInMillimeter;
    } Config;
};

enum class ComputeMode : uint8_t {
    eHost = 0,
    eDeviceAccelerated = 1,
    // Special mode that selects host vs device implementation at compile time.
    eAuto = std::numeric_limits<uint8_t>::max()
};

constexpr ComputeMode AutoDetectComputeMode() {
#ifdef APP_USE_CUDA
    return ComputeMode::eDeviceAccelerated;
#else
    return ComputeMode::eHost;
#endif
}

class IDepthMapToBucketComputer {
public:
    virtual ~IDepthMapToBucketComputer() = default;

    virtual void Compute(const DepthMapToBucketComputeInfo &info) = 0;

    static std::unique_ptr<IDepthMapToBucketComputer> Create(ComputeMode mode);
};

std::optional<uint32_t> GetLowestProportionFromDepthMap(IDepthMapToBucketComputer &computer,
                                                        std::vector<uint32_t> &scratchBuckets,
                                                        const LowestProportionFromDepthMapComputeInfo &info);
// Returns:
// - std::nullopt when there are no valid depth pixels within the configured [DistanceMin, DistanceMax] range.
// - Otherwise returns the distance in millimeters where the cumulative histogram proportion is reached.
