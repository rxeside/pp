#pragma once
#include "IImageFilter.h"
#include <CL/cl.h>
#include <string>
#include <vector>

class MotionBlurFilter : public IImageFilter
{
public:
    MotionBlurFilter(int initialBlurLength);
    ~MotionBlurFilter() override;

    void ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels) override;
    void SetEffectRadius(int blurLength) override; // Здесь radius - это длина размытия
    std::string GetName() const override { return "Motion Blur (Horizontal)"; }

private:
    void InitializeOpenCl();
    void ReleaseOpenCl();
    void CreateKernel();

    int m_blurLength;

    cl_device_id m_deviceId = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_commandQueue = nullptr;
    cl_program m_program = nullptr;
    cl_kernel m_kernel = nullptr;

    static const std::string m_kernelSource;
};