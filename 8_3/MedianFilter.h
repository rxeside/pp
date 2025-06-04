#pragma once
#include "IImageFilter.h"
#include <CL/cl.h>
#include <string>
#include <vector>

class MedianFilter : public IImageFilter
{
public:
    MedianFilter(int initialRadius);
    ~MedianFilter() override;

    void ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels) override;
    void SetEffectRadius(int radius) override;
    std::string GetName() const override { return "Median Filter"; }

private:
    void InitializeOpenCl();
    void ReleaseOpenCl();
    void CreateKernel();

    int m_effectRadius;
    static constexpr int MAX_KERNEL_SUPPORTED_RADIUS = 10;


    cl_device_id m_deviceId = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_commandQueue = nullptr;
    cl_program m_program = nullptr;
    cl_kernel m_kernel = nullptr;

    static const std::string m_kernelSource;
};