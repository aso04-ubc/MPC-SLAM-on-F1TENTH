#include "DeviceComputation.cuh"

#include <cstdio>

__global__ void ComputeDepthBucketKernel(ComputeDepthBucketKernelSubmissionInfo info) {
    // Debug output from first thread
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[CUDA Kernel] Launched! ROI: X[%u,%u) Y[%u,%u), DepthMap: %ux%u, BucketStep: %u, BucketCount: %u\n",
    //            info.Configuration.RegionOfInterestStartX,
    //            info.Configuration.RegionOfInterestEndX,
    //            info.Configuration.RegionOfInterestStartY,
    //            info.Configuration.RegionOfInterestEndY,
    //            info.DeviceVisible.Read.DepthMapWidth,
    //            info.DeviceVisible.Read.DepthMapHeight,
    //            info.Configuration.BucketStepSizeInMillimeter,
    //            info.DeviceVisible.Write.BucketCount);
    // }

    uint32_t currentPixelX = info.Configuration.RegionOfInterestStartX + (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t currentPixelY = info.Configuration.RegionOfInterestStartY + (blockIdx.y * blockDim.y + threadIdx.y);

    if (currentPixelX >= info.Configuration.RegionOfInterestEndX ||
        currentPixelY >= info.Configuration.RegionOfInterestEndY ||
        currentPixelX >= info.DeviceVisible.Read.DepthMapWidth ||
        currentPixelY >= info.DeviceVisible.Read.DepthMapHeight) {
        return;
    }

    uint32_t linearPixelIndex = currentPixelY * info.DeviceVisible.Read.DepthMapWidth + currentPixelX;
    uint16_t depthValueInMillimeter = info.DeviceVisible.Read.DepthMapInMillimeterBuffer[linearPixelIndex];

    // Debug: Print first few valid depth values
    // if (depthValueInMillimeter > 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4 && threadIdx.y < 4) {
    //     printf("[CUDA Thread %u,%u] Pixel(%u,%u) depth=%u\n",
    //            threadIdx.x, threadIdx.y, currentPixelX, currentPixelY, depthValueInMillimeter);
    // }

    if (depthValueInMillimeter == 0) {
        return;
    }

    uint32_t targetBucketIndex = (uint32_t)depthValueInMillimeter / info.Configuration.BucketStepSizeInMillimeter;

    if (targetBucketIndex < info.DeviceVisible.Write.BucketCount) {
        atomicAdd(&info.DeviceVisible.Write.BucketBuffer[targetBucketIndex], 1);
    }
}

void ComputeDepthBucketKernelLaunch(ComputeDepthBucketKernelSubmissionInfo info, dim3 blocksPerGrid,
    dim3 threadsPerBlock) {
    ComputeDepthBucketKernel<<<blocksPerGrid, threadsPerBlock>>>(info);
}