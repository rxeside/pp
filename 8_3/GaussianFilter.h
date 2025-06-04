#pragma once
#include "IImageFilter.h"
#include <CL/cl.h> // C API
#include <string>
#include <vector>

class GaussianFilter : public IImageFilter
{
public:
    explicit GaussianFilter(int initialRadius); // Контекст OpenCL будет создан внутри
    ~GaussianFilter() override;

    void ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels) override;
    void SetEffectRadius(int radius) override;
    [[nodiscard]] std::string GetName() const override { return "Gaussian Blur"; }

private:
    void InitializeOpenCl();
    void ReleaseOpenCl();
    void CreateKernels(); // Создает оба ядра
    static std::vector<float> CreateGaussianKernelValues(int radius, float sigma);

    int m_effectRadius;

    cl_device_id m_deviceId = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_commandQueue = nullptr;
    cl_program m_program = nullptr;
    cl_kernel m_blurPassKernel = nullptr;
    cl_kernel m_transposeKernel = nullptr;

    static const std::string m_blurPassKernelSource;
    static const std::string m_transposeKernelSource;
};