#include "MotionBlurFilter.h"
#include "OpenCLUtils.h"
#include <iostream>
#include <algorithm> // For std::max

const std::string MotionBlurFilter::m_kernelSource = R"CLC(
__kernel void ApplyMotionBlur(
    __global const uchar* inputImage,
    __global uchar* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int numChannels,
    const int blurLength) // Длина следа размытия
{
    int globalX = get_global_id(0);
    int globalY = get_global_id(1);

    if (globalX >= imageWidth || globalY >= imageHeight) return;

    for (int c = 0; c < numChannels; ++c) {
        float accumulatedColor = 0.0f;
        int samplesCount = 0;

        // Горизонтальное размытие. Для других углов нужна другая логика ядра.
        // blurLength определяет, сколько пикселей усреднять.
        int startOffset = -blurLength / 2;
        int endOffset = blurLength / 2;
        // Для четной длины, чтобы было симметрично, можно сделать так:
        // (blurLength=4 -> -1,0,1,2 или -2,-1,0,1). Оставим как есть для простоты: -L/2 .. L/2
        // Если blurLength=1, то start=0, end=0, т.е. только текущий пиксель.
        if (blurLength == 1) { startOffset = 0; endOffset = 0; }
        else if (blurLength % 2 == 0 && blurLength > 0) { // Для четной длины, чтобы было симметрично вокруг пикселя
             endOffset = blurLength / 2 -1; // Например, для 4: -2, -1, 0, 1
        }


        for (int offset = startOffset; offset <= endOffset; ++offset) {
            int sampleX = clamp(globalX + offset, 0, imageWidth - 1);
            int sampleIndex = (globalY * imageWidth + sampleX) * numChannels + c;

            accumulatedColor += (float)inputImage[sampleIndex];
            samplesCount++;
        }

        int outputIndex = (globalY * imageWidth + globalX) * numChannels + c;
        if (samplesCount > 0) {
            outputImage[outputIndex] = (uchar)(accumulatedColor / samplesCount);
        } else {
            // Это может случиться, если blurLength = 0
            outputImage[outputIndex] = inputImage[outputIndex];
        }
    }
}
)CLC";

MotionBlurFilter::MotionBlurFilter(int initialBlurLength)
        : m_blurLength(initialBlurLength)
{
    InitializeOpenCl();
    m_program = CreateProgramWithSource(m_context, m_deviceId, m_kernelSource);
    CreateKernel();
}

MotionBlurFilter::~MotionBlurFilter()
{
    ReleaseOpenCl();
}

void MotionBlurFilter::InitializeOpenCl()
{
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    CheckCLError(err, "clGetPlatformIDs (count)");
    if (numPlatforms == 0) throw std::runtime_error("MotionBlurFilter: No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (list)");

    cl_platform_id platform = platforms[0];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_deviceId, nullptr);
    if (err == CL_DEVICE_NOT_FOUND || m_deviceId == nullptr) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &m_deviceId, nullptr);
        CheckCLError(err, "clGetDeviceIDs (CPU) for MotionBlur");
    } else {
        CheckCLError(err, "clGetDeviceIDs (GPU) for MotionBlur");
    }

    m_context = clCreateContext(nullptr, 1, &m_deviceId, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext for MotionBlur");

#if defined(CL_VERSION_2_0) && CL_TARGET_OPENCL_VERSION >= 200
    m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_deviceId, 0, &err);
#else
    m_commandQueue = clCreateCommandQueue(m_context, m_deviceId, 0, &err);
#endif
    CheckCLError(err, "clCreateCommandQueue for MotionBlur");
}

void MotionBlurFilter::CreateKernel() {
    cl_int err;
    m_kernel = clCreateKernel(m_program, "ApplyMotionBlur", &err);
    CheckCLError(err, "clCreateKernel (ApplyMotionBlur)");
}

void MotionBlurFilter::ReleaseOpenCl()
{
    if (m_kernel) clReleaseKernel(m_kernel);
    if (m_program) clReleaseProgram(m_program);
    if (m_commandQueue) clReleaseCommandQueue(m_commandQueue);
    if (m_context) clReleaseContext(m_context);
}

void MotionBlurFilter::ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels)
{
    if (m_blurLength <= 0) return; // Длина 0 или 1 обычно означает отсутствие эффекта или минимальный.
    // Если m_blurLength = 1, ядро возьмет только текущий пиксель.

    cl_int err;
    size_t imageSizeBytes = static_cast<size_t>(width) * height * channels * sizeof(unsigned char);

    cl_mem inputBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        imageSizeBytes, imageData.data(), &err);
    CheckCLError(err, "MotionBlur clCreateBuffer (inputBuffer)");
    cl_mem outputBuffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY,
                                         imageSizeBytes, nullptr, &err);
    CheckCLError(err, "MotionBlur clCreateBuffer (outputBuffer)");

    err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &inputBuffer); CheckCLError(err, "MotionBlur SetArg 0");
    err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &outputBuffer); CheckCLError(err, "MotionBlur SetArg 1");
    err = clSetKernelArg(m_kernel, 2, sizeof(int), &width); CheckCLError(err, "MotionBlur SetArg 2");
    err = clSetKernelArg(m_kernel, 3, sizeof(int), &height); CheckCLError(err, "MotionBlur SetArg 3");
    err = clSetKernelArg(m_kernel, 4, sizeof(int), &channels); CheckCLError(err, "MotionBlur SetArg 4");
    err = clSetKernelArg(m_kernel, 5, sizeof(int), &m_blurLength); CheckCLError(err, "MotionBlur SetArg 5");

    size_t globalWorkSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    err = clEnqueueNDRangeKernel(m_commandQueue, m_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "MotionBlur clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(m_commandQueue, outputBuffer, CL_TRUE, 0,
                              imageSizeBytes, imageData.data(), 0, nullptr, nullptr);
    CheckCLError(err, "MotionBlur clEnqueueReadBuffer");

    clFinish(m_commandQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
}

void MotionBlurFilter::SetEffectRadius(int blurLength) // radius является blurLength
{
    m_blurLength = std::max(0, blurLength);
}