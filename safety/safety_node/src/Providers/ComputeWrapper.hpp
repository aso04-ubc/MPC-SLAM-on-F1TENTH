#pragma once

#include <cstdint>
#include <memory>


struct DepthMapInfo {
    const uint16_t *Buffer;
    uint32_t Width;
    uint32_t Height;
};

struct BucketInfo {
    uint32_t *Buffer;
    uint32_t Count;
    uint32_t StepSizeInMillimeter;
};

struct RegionOfInterestInfo {
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
