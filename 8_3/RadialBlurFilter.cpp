#include "RadialBlurFilter.h"
#include "OpenCLUtils.h"
#include <iostream>
#include <cmath>     // Для sqrt
#include <algorithm> // Для std::max

const std::string RadialBlurFilter::m_kernelSource = R"CLC(
__kernel void ApplyRadialBlur(
    __global const uchar* inputImage,
    __global uchar* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int numChannels,
    const int blurIntensity) // Интенсивность / количество сэмплов
{
    int globalX = get_global_id(0);
    int globalY = get_global_id(1);

    if (globalX >= imageWidth || globalY >= imageHeight) return;

    float centerX = (float)imageWidth / 2.0f;
    float centerY = (float)imageHeight / 2.0f;

    float deltaX = (float)globalX - centerX;
    float deltaY = (float)globalY - centerY;
    float distanceToCenter = sqrt(deltaX * deltaX + deltaY * deltaY);

    if (distanceToCenter < 1.0f || blurIntensity == 0) {
        for (int ch = 0; ch < numChannels; ++ch) {
            int currentIndex = (globalY * imageWidth + globalX) * numChannels + ch;
            outputImage[currentIndex] = inputImage[currentIndex];
        }
        return;
    }

    float dirX = deltaX / distanceToCenter;
    float dirY = deltaY / distanceToCenter;

    // Максимальное расстояние до центра (примерно половина диагонали)
    // Используется для масштабирования эффекта в зависимости от удаленности от центра.
    float maxPossibleDist = 0.5f * sqrt((float)(imageWidth * imageWidth + imageHeight * imageHeight));
    if (maxPossibleDist < 1.0f) maxPossibleDist = 1.0f; // Избегаем деления на ноль

    // sampleStep определяет, насколько далеко друг от друга берутся сэмплы.
    // Увеличивается с интенсивностью и расстоянием от центра.
    // Этот коэффициент эмпирический, можно подбирать.
    float stepFactor = 0.005f * blurIntensity; // Уменьшил, чтобы не было слишком сильно
    float sampleStep = 1.0f + (distanceToCenter / maxPossibleDist) * stepFactor * blurIntensity;
    sampleStep = max(1.0f, sampleStep);

    int numSamples = max(1, blurIntensity / 2 + 1); // Количество сэмплов, можно тоже связать с blurIntensity

    for (int c = 0; c < numChannels; ++c) {
        float accumulatedColor = 0.0f;
        int actualSamplesCount = 0;

        // Сэмплируем вдоль радиальной линии
        // Для симметричного размытия можно сэмплировать от -numSamples/2 до +numSamples/2
        for (int s = 0; s < numSamples; ++s) {
            // float currentOffset = ((float)s - (float)(numSamples-1)/2.0f) * sampleStep; // Для симметричного
            float currentOffset = (float)s * sampleStep; // Сэмплируем от текущего пикселя "назад" к центру

            int sampleX = clamp((int)((float)globalX - dirX * currentOffset), 0, imageWidth - 1);
            int sampleY = clamp((int)((float)globalY - dirY * currentOffset), 0, imageHeight - 1);

            int sampleIndex = (sampleY * imageWidth + sampleX) * numChannels + c;
            accumulatedColor += (float)inputImage[sampleIndex];
            actualSamplesCount++;
        }

        int outputIndex = (globalY * imageWidth + globalX) * numChannels + c;
        if (actualSamplesCount > 0) {
            outputImage[outputIndex] = (uchar)(accumulatedColor / actualSamplesCount);
        } else {
            outputImage[outputIndex] = inputImage[outputIndex];
        }
    }
}
)CLC";

RadialBlurFilter::RadialBlurFilter(int initialIntensity)
        : m_intensity(initialIntensity)
{
    InitializeOpenCl();
    m_program = CreateProgramWithSource(m_context, m_deviceId, m_kernelSource);
    CreateKernel();
}

RadialBlurFilter::~RadialBlurFilter()
{
    ReleaseOpenCl();
}

void RadialBlurFilter::InitializeOpenCl()
{
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    CheckCLError(err, "clGetPlatformIDs (count)");
    if (numPlatforms == 0) throw std::runtime_error("RadialBlurFilter: No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (list)");

    cl_platform_id platform = platforms[0];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_deviceId, nullptr);
    if (err == CL_DEVICE_NOT_FOUND || m_deviceId == nullptr) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &m_deviceId, nullptr);
        CheckCLError(err, "clGetDeviceIDs (CPU) for RadialBlur");
    } else {
        CheckCLError(err, "clGetDeviceIDs (GPU) for RadialBlur");
    }

    m_context = clCreateContext(nullptr, 1, &m_deviceId, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext for RadialBlur");

#if defined(CL_VERSION_2_0) && CL_TARGET_OPENCL_VERSION >= 200
    m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_deviceId, 0, &err);
#else
    m_commandQueue = clCreateCommandQueue(m_context, m_deviceId, 0, &err);
#endif
    CheckCLError(err, "clCreateCommandQueue for RadialBlur");
}

void RadialBlurFilter::CreateKernel() {
    cl_int err;
    m_kernel = clCreateKernel(m_program, "ApplyRadialBlur", &err);
    CheckCLError(err, "clCreateKernel (ApplyRadialBlur)");
}

void RadialBlurFilter::ReleaseOpenCl()
{
    if (m_kernel) clReleaseKernel(m_kernel);
    if (m_program) clReleaseProgram(m_program);
    if (m_commandQueue) clReleaseCommandQueue(m_commandQueue);
    if (m_context) clReleaseContext(m_context);
}

void RadialBlurFilter::ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels)
{
    if (m_intensity <= 0) return; // Интенсивность 0 - нет эффекта

    cl_int err;
    size_t imageSizeBytes = static_cast<size_t>(width) * height * channels * sizeof(unsigned char);

    cl_mem inputBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        imageSizeBytes, imageData.data(), &err);
    CheckCLError(err, "RadialBlur clCreateBuffer (inputBuffer)");
    cl_mem outputBuffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY,
                                         imageSizeBytes, nullptr, &err);
    CheckCLError(err, "RadialBlur clCreateBuffer (outputBuffer)");

    err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &inputBuffer); CheckCLError(err, "RadialBlur SetArg 0");
    err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &outputBuffer); CheckCLError(err, "RadialBlur SetArg 1");
    err = clSetKernelArg(m_kernel, 2, sizeof(int), &width); CheckCLError(err, "RadialBlur SetArg 2");
    err = clSetKernelArg(m_kernel, 3, sizeof(int), &height); CheckCLError(err, "RadialBlur SetArg 3");
    err = clSetKernelArg(m_kernel, 4, sizeof(int), &channels); CheckCLError(err, "RadialBlur SetArg 4");
    err = clSetKernelArg(m_kernel, 5, sizeof(int), &m_intensity); CheckCLError(err, "RadialBlur SetArg 5");

    size_t globalWorkSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    err = clEnqueueNDRangeKernel(m_commandQueue, m_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "RadialBlur clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(m_commandQueue, outputBuffer, CL_TRUE, 0,
                              imageSizeBytes, imageData.data(), 0, nullptr, nullptr);
    CheckCLError(err, "RadialBlur clEnqueueReadBuffer");

    clFinish(m_commandQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
}

void RadialBlurFilter::SetEffectRadius(int intensity) // radius является intensity
{
    m_intensity = std::max(0, intensity);
}