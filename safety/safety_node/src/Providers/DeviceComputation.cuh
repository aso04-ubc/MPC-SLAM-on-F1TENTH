#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct ComputeDepthBucketKernelSubmissionInfo {
    struct {
        struct {
            const uint16_t *DepthMapInMillimeterBuffer;
            uint32_t DepthMapWidth;
            uint32_t DepthMapHeight;
        } Read;

        struct {
            uint32_t *BucketBuffer;
            uint32_t BucketCount;
        } Write;
    } DeviceVisible;

    struct {
        uint32_t RegionOfInterestStartX;
        uint32_t RegionOfInterestEndX;
        uint32_t RegionOfInterestStartY;
        uint32_t RegionOfInterestEndY;
        uint32_t BucketStepSizeInMillimeter;
    } Configuration;
};

void ComputeDepthBucketKernelLaunch(
    ComputeDepthBucketKernelSubmissionInfo info,
    dim3 blocksPerGrid,
    dim3 threadsPerBlock
);