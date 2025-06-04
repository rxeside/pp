#include "MedianFilter.h"
#include "OpenCLUtils.h"
#include <iostream>
#include <algorithm> // For std::min, std::max


const std::string MedianFilter::m_kernelSource = R"CLC(
// Простая пузырьковая сортировка для небольшого массива
void SortWindowSegment(uchar* segment, int count) {
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (segment[j] > segment[j + 1]) {
                uchar temp = segment[j];
                segment[j] = segment[j + 1];
                segment[j + 1] = temp;
            }
        }
    }
}

__kernel void ApplyMedianFilter(
    __global const uchar* inputImage,
    __global uchar* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int numChannels,
    const int filterRadius)
{
    int globalX = get_global_id(0);
    int globalY = get_global_id(1);

    if (globalX >= imageWidth || globalY >= imageHeight) return;

    // Максимальный размер окна (2*10+1)*(2*10+1) = 441 для радиуса 10.
    // Если filterRadius больше, это приведет к проблемам.
    // Проверка на стороне CPU должна ограничивать filterRadius.
    uchar windowValues[441]; // Убедитесь, что это достаточно для MAX_KERNEL_SUPPORTED_RADIUS

    int windowDimension = 2 * filterRadius + 1;
    // int windowPixelCount = windowDimension * windowDimension; // Не используется явно

    for (int c = 0; c < numChannels; ++c) { // Обрабатываем каждый канал отдельно
        int currentPixelCountInWindow = 0;
        for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
            for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX) {
                int sampleX = clamp(globalX + offsetX, 0, imageWidth - 1);
                int sampleY = clamp(globalY + offsetY, 0, imageHeight - 1);

                int sampleIndex = (sampleY * imageWidth + sampleX) * numChannels + c;
                if (currentPixelCountInWindow < 441) { // Защита от переполнения windowValues
                   windowValues[currentPixelCountInWindow++] = inputImage[sampleIndex];
                }
            }
        }

        SortWindowSegment(windowValues, currentPixelCountInWindow);

        int outputIndex = (globalY * imageWidth + globalX) * numChannels + c;
        if (currentPixelCountInWindow > 0) {
            outputImage[outputIndex] = windowValues[currentPixelCountInWindow / 2]; // Медиана
        } else {
            // Этого не должно случиться, если filterRadius >= 0
             outputImage[outputIndex] = inputImage[outputIndex];
        }
    }
}
)CLC";

MedianFilter::MedianFilter(int initialRadius)
        : m_effectRadius(initialRadius)
{
    InitializeOpenCl();
    m_program = CreateProgramWithSource(m_context, m_deviceId, m_kernelSource);
    CreateKernel();
}

MedianFilter::~MedianFilter()
{
    ReleaseOpenCl();
}

void MedianFilter::InitializeOpenCl()
{
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    CheckCLError(err, "clGetPlatformIDs (count)");
    if (numPlatforms == 0) throw std::runtime_error("MedianFilter: No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (list)");

    cl_platform_id platform = platforms[0];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_deviceId, nullptr);
    if (err == CL_DEVICE_NOT_FOUND || m_deviceId == nullptr) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &m_deviceId, nullptr);
        CheckCLError(err, "clGetDeviceIDs (CPU) for MedianFilter");
    } else {
        CheckCLError(err, "clGetDeviceIDs (GPU) for MedianFilter");
    }

    m_context = clCreateContext(nullptr, 1, &m_deviceId, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext for MedianFilter");

#if defined(CL_VERSION_2_0) && CL_TARGET_OPENCL_VERSION >= 200
    m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_deviceId, 0, &err);
#else
    m_commandQueue = clCreateCommandQueue(m_context, m_deviceId, 0, &err);
#endif
    CheckCLError(err, "clCreateCommandQueue for MedianFilter");
}

void MedianFilter::CreateKernel() {
    cl_int err;
    m_kernel = clCreateKernel(m_program, "ApplyMedianFilter", &err);
    CheckCLError(err, "clCreateKernel (ApplyMedianFilter)");
}

void MedianFilter::ReleaseOpenCl()
{
    if (m_kernel) clReleaseKernel(m_kernel);
    if (m_program) clReleaseProgram(m_program);
    if (m_commandQueue) clReleaseCommandQueue(m_commandQueue);
    if (m_context) clReleaseContext(m_context);
}

void MedianFilter::ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels)
{
    // Радиус 0 для медианного фильтра означает окно 1x1, т.е. без изменений.
    // Однако, если m_effectRadius = 0, то windowDimension = 1, windowPixelCount = 1.
    // Это корректно вернет исходный пиксель.

    // Ограничиваем радиус тем, что поддерживает ядро
    int actualRadius = std::min(m_effectRadius, MAX_KERNEL_SUPPORTED_RADIUS);
    if (m_effectRadius > MAX_KERNEL_SUPPORTED_RADIUS) {
        std::cout << "Warning: MedianFilter radius " << m_effectRadius
                  << " capped at " << MAX_KERNEL_SUPPORTED_RADIUS
                  << " due to kernel limitations." << std::endl;
    }
    if (actualRadius < 0) actualRadius = 0; // Не должно быть, но на всякий случай


    cl_int err;
    size_t imageSizeBytes = static_cast<size_t>(width) * height * channels * sizeof(unsigned char);

    cl_mem inputBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        imageSizeBytes, imageData.data(), &err);
    CheckCLError(err, "MedianFilter clCreateBuffer (inputBuffer)");
    cl_mem outputBuffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY,
                                         imageSizeBytes, nullptr, &err);
    CheckCLError(err, "MedianFilter clCreateBuffer (outputBuffer)");

    err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &inputBuffer); CheckCLError(err, "Median SetArg 0");
    err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &outputBuffer); CheckCLError(err, "Median SetArg 1");
    err = clSetKernelArg(m_kernel, 2, sizeof(int), &width); CheckCLError(err, "Median SetArg 2");
    err = clSetKernelArg(m_kernel, 3, sizeof(int), &height); CheckCLError(err, "Median SetArg 3");
    err = clSetKernelArg(m_kernel, 4, sizeof(int), &channels); CheckCLError(err, "Median SetArg 4");
    err = clSetKernelArg(m_kernel, 5, sizeof(int), &actualRadius); CheckCLError(err, "Median SetArg 5");

    size_t globalWorkSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    err = clEnqueueNDRangeKernel(m_commandQueue, m_kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "MedianFilter clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(m_commandQueue, outputBuffer, CL_TRUE, 0,
                              imageSizeBytes, imageData.data(), 0, nullptr, nullptr);
    CheckCLError(err, "MedianFilter clEnqueueReadBuffer");

    clFinish(m_commandQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
}

void MedianFilter::SetEffectRadius(int radius)
{
    m_effectRadius = std::max(0, radius);
}