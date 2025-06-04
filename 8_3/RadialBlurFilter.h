#pragma once
#include "IImageFilter.h"
#include <CL/cl.h>
#include <string>
#include <vector>

class RadialBlurFilter : public IImageFilter
{
public:
    RadialBlurFilter(int initialIntensity);
    ~RadialBlurFilter() override;

    void ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels) override;
    void SetEffectRadius(int intensity) override; // radius - это интенсивность/количество сэмплов
    std::string GetName() const override { return "Radial Blur"; }

private:
    void InitializeOpenCl();
    void ReleaseOpenCl();
    void CreateKernel();

    int m_intensity; // Интенсивность размытия / количество сэмплов

    cl_device_id m_deviceId = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_commandQueue = nullptr;
    cl_program m_program = nullptr;
    cl_kernel m_kernel = nullptr;

    static const std::string m_kernelSource;
};