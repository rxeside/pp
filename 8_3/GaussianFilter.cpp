#include "GaussianFilter.h"
#include "OpenCLUtils.h"
#include <cmath>
#include <iostream>
#include <algorithm> // For std::clamp, std::max

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const std::string GaussianFilter::m_blurPassKernelSource = R"CLC(
// Работает с uchar4, т.е. 4 канала на пиксель.
__kernel void BlurPass(
    __global const uchar4* inputImage, // Ожидает данные в формате RGBA
    __global uchar4* outputImage,
    __constant float* filterKernel,
    const int kernelRadius,
    const int imageWidth, // Ширина текущего измерения (может быть height после transpose)
    const int imageHeight) // Высота текущего измерения (может быть width после transpose)
{
    int gid = get_global_id(0);
    if (gid >= imageWidth * imageHeight) return; // Общее количество пикселей в текущем измерении

    int currentX = gid % imageWidth; // Координата вдоль размываемого направления
    int currentY = gid / imageWidth; // Координата перпендикулярная размываемому направлению

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int offset = -kernelRadius; offset <= kernelRadius; ++offset)
    {
        int sampleCoord = clamp(currentX + offset, 0, imageWidth - 1);
        uchar4 pixelColor = inputImage[currentY * imageWidth + sampleCoord];

        float4 floatPixelColor = convert_float4(pixelColor);

        float weight = filterKernel[offset + kernelRadius];
        sum += floatPixelColor * weight;
    }
    // sum.w = (float)inputImage[gid].s3; // Вариант: сохранить исходную альфу
    outputImage[gid] = convert_uchar4_sat_rte(sum);
}
)CLC";

const std::string GaussianFilter::m_transposeKernelSource = R"CLC(
__kernel void TransposeImage(
    __global const uchar4* inputImage,
    __global uchar4* outputImage,
    const int imageWidth, // Оригинальная ширина
    const int imageHeight) // Оригинальная высота
{
    int currentX = get_global_id(0); // Становится новой Y координатой
    int currentY = get_global_id(1); // Становится новой X координатой

    if (currentX >= imageWidth || currentY >= imageHeight) return;

    outputImage[currentX * imageHeight + currentY] = inputImage[currentY * imageWidth + currentX];
}
)CLC";


GaussianFilter::GaussianFilter(int initialRadius)
        : m_effectRadius(initialRadius)
{
    InitializeOpenCl();
    m_program = CreateProgramWithSource(m_context, m_deviceId, m_blurPassKernelSource + m_transposeKernelSource);
    CreateKernels();
}

GaussianFilter::~GaussianFilter()
{
    ReleaseOpenCl();
}

void GaussianFilter::InitializeOpenCl()
{
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(1, nullptr, &numPlatforms); // Проверяем, есть ли хотя бы одна
    if (numPlatforms == 0) clGetPlatformIDs(0, nullptr, &numPlatforms); // Если нет, получаем точное число
    CheckCLError(err, "clGetPlatformIDs (count)");
    if (numPlatforms == 0) throw std::runtime_error("GaussianFilter: No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (list)");

    cl_platform_id platform = platforms[0];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_deviceId, nullptr);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &m_deviceId, nullptr);
        CheckCLError(err, "clGetDeviceIDs (CPU)");
    } else {
        CheckCLError(err, "clGetDeviceIDs (GPU)");
    }

    m_context = clCreateContext(nullptr, 1, &m_deviceId, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext");

#if defined(CL_VERSION_2_0)
    m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_deviceId, nullptr, &err);
#else
    m_commandQueue = clCreateCommandQueue(m_context, m_deviceId, 0, &err);
#endif
    CheckCLError(err, "clCreateCommandQueue");
}

void GaussianFilter::CreateKernels() {
    cl_int err;
    m_blurPassKernel = clCreateKernel(m_program, "BlurPass", &err);
    CheckCLError(err, "clCreateKernel (BlurPass)");
    m_transposeKernel = clCreateKernel(m_program, "TransposeImage", &err);
    CheckCLError(err, "clCreateKernel (TransposeImage)");
}

void GaussianFilter::ReleaseOpenCl()
{
    if (m_blurPassKernel) clReleaseKernel(m_blurPassKernel);
    if (m_transposeKernel) clReleaseKernel(m_transposeKernel);
    if (m_program) clReleaseProgram(m_program);
    if (m_commandQueue) clReleaseCommandQueue(m_commandQueue);
    if (m_context) clReleaseContext(m_context);
}

std::vector<float> GaussianFilter::CreateGaussianKernelValues(int radius, float sigma)
{
    int kernelSize = 2 * radius + 1;
    std::vector<float> kernel(kernelSize);
    float sum = 0.0f;

    for (int i = -radius; i <= radius; ++i)
    {
        float value = std::exp(-static_cast<float>(i * i) / (2.0f * sigma * sigma));
        // Деление на (std::sqrt(2.0f * M_PI) * sigma) можно опустить, т.к. будет нормализация
        kernel[i + radius] = value;
        sum += value;
    }
    if (sum > 0) { // Избегаем деления на ноль, если все значения ядра нулевые (маловероятно)
        for (float& val : kernel) val /= sum;
    }
    return kernel;
}

void GaussianFilter::ApplyFilter(std::vector<unsigned char>& imageData, int width, int height, int channels)
{
    if (m_effectRadius == 0) return;
    // Фильтр ожидает 4 канала (uchar4). Если на входе 3, нужно преобразовать.
    // Для простоты, сейчас будем предполагать, что channels == 4.
    if (channels != 4) {
        std::cerr << "GaussianFilter current implementation expects 4 channels (RGBA)." << std::endl;
        // Можно добавить конвертацию RGB -> RGBA здесь или выбросить исключение
        // Например, создать новый std::vector<unsigned char> с 4 каналами.
        // Пока просто выйдем.
        return;
    }

    cl_int err;
    size_t numPixels = static_cast<size_t>(width) * height;
    size_t imageSizeBytes = numPixels * channels * sizeof(unsigned char); // channels здесь всегда 4

    float sigma = std::max(1.0f, static_cast<float>(m_effectRadius) / 2.0f);
    std::vector<float> gaussianKernelVec = CreateGaussianKernelValues(m_effectRadius, sigma);
    size_t kernelSizeBytes = gaussianKernelVec.size() * sizeof(float);

    cl_mem inputOutputBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              imageSizeBytes, imageData.data(), &err);
    CheckCLError(err, "clCreateBuffer (inputOutputBuffer)");
    cl_mem tempBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, imageSizeBytes, nullptr, &err);
    CheckCLError(err, "clCreateBuffer (tempBuffer)");
    cl_mem kernelCLBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           kernelSizeBytes, gaussianKernelVec.data(), &err);
    CheckCLError(err, "clCreateBuffer (kernelCLBuffer)");

    // --- Горизонтальный проход ---
    err = clSetKernelArg(m_blurPassKernel, 0, sizeof(cl_mem), &inputOutputBuffer); CheckCLError(err, "SetArg Blur 0");
    err = clSetKernelArg(m_blurPassKernel, 1, sizeof(cl_mem), &tempBuffer);        CheckCLError(err, "SetArg Blur 1");
    err = clSetKernelArg(m_blurPassKernel, 2, sizeof(cl_mem), &kernelCLBuffer);    CheckCLError(err, "SetArg Blur 2");
    err = clSetKernelArg(m_blurPassKernel, 3, sizeof(int), &m_effectRadius);       CheckCLError(err, "SetArg Blur 3");
    err = clSetKernelArg(m_blurPassKernel, 4, sizeof(int), &width);                CheckCLError(err, "SetArg Blur 4");
    err = clSetKernelArg(m_blurPassKernel, 5, sizeof(int), &height);               CheckCLError(err, "SetArg Blur 5");

    size_t globalWorkSizePass1[1] = { numPixels }; // Одномерное ядро
    err = clEnqueueNDRangeKernel(m_commandQueue, m_blurPassKernel, 1, nullptr, globalWorkSizePass1, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "EnqueueNDRangeKernel (BlurPass Horizontal)");

    // --- Транспонирование 1 (tempBuffer -> inputOutputBuffer) ---
    err = clSetKernelArg(m_transposeKernel, 0, sizeof(cl_mem), &tempBuffer);         CheckCLError(err, "SetArg Transpose1 0");
    err = clSetKernelArg(m_transposeKernel, 1, sizeof(cl_mem), &inputOutputBuffer);  CheckCLError(err, "SetArg Transpose1 1");
    err = clSetKernelArg(m_transposeKernel, 2, sizeof(int), &width);                 CheckCLError(err, "SetArg Transpose1 2");
    err = clSetKernelArg(m_transposeKernel, 3, sizeof(int), &height);                CheckCLError(err, "SetArg Transpose1 3");

    size_t globalWorkSizeTranspose[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    err = clEnqueueNDRangeKernel(m_commandQueue, m_transposeKernel, 2, nullptr, globalWorkSizeTranspose, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "EnqueueNDRangeKernel (Transpose1)");

    // --- Вертикальный проход (на транспонированном изображении, inputOutputBuffer -> tempBuffer) ---
    // Размеры для ядра размытия теперь height (новая ширина) и width (новая высота)
    int transposedWidth = height;
    int transposedHeight = width;
    err = clSetKernelArg(m_blurPassKernel, 0, sizeof(cl_mem), &inputOutputBuffer); CheckCLError(err, "SetArg BlurV 0");
    err = clSetKernelArg(m_blurPassKernel, 1, sizeof(cl_mem), &tempBuffer);        CheckCLError(err, "SetArg BlurV 1");
    // Arg 2 (kernelCLBuffer) и 3 (m_effectRadius) остаются теми же
    err = clSetKernelArg(m_blurPassKernel, 4, sizeof(int), &transposedWidth);      CheckCLError(err, "SetArg BlurV 4");
    err = clSetKernelArg(m_blurPassKernel, 5, sizeof(int), &transposedHeight);     CheckCLError(err, "SetArg BlurV 5");

    // globalWorkSizePass1 (numPixels) остается тем же, т.к. количество пикселей не изменилось
    err = clEnqueueNDRangeKernel(m_commandQueue, m_blurPassKernel, 1, nullptr, globalWorkSizePass1, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "EnqueueNDRangeKernel (BlurPass Vertical)");

    // --- Транспонирование 2 (обратно, tempBuffer -> inputOutputBuffer) ---
    // Размеры для ядра транспонирования теперь transposedWidth=height, transposedHeight=width
    err = clSetKernelArg(m_transposeKernel, 0, sizeof(cl_mem), &tempBuffer);         CheckCLError(err, "SetArg Transpose2 0");
    err = clSetKernelArg(m_transposeKernel, 1, sizeof(cl_mem), &inputOutputBuffer);  CheckCLError(err, "SetArg Transpose2 1");
    err = clSetKernelArg(m_transposeKernel, 2, sizeof(int), &transposedWidth);       CheckCLError(err, "SetArg Transpose2 2"); // Старая ширина транспонированного = новая высота исходного
    err = clSetKernelArg(m_transposeKernel, 3, sizeof(int), &transposedHeight);      CheckCLError(err, "SetArg Transpose2 3"); // Старая высота транспонированного = новая ширина исходного

    size_t globalWorkSizeTransposeBack[2] = {static_cast<size_t>(transposedWidth), static_cast<size_t>(transposedHeight)}; // (height, width)
    err = clEnqueueNDRangeKernel(m_commandQueue, m_transposeKernel, 2, nullptr, globalWorkSizeTransposeBack, nullptr, 0, nullptr, nullptr);
    CheckCLError(err, "EnqueueNDRangeKernel (Transpose2)");

    // Чтение результата
    err = clEnqueueReadBuffer(m_commandQueue, inputOutputBuffer, CL_TRUE, 0, imageSizeBytes, imageData.data(), 0, nullptr, nullptr);
    CheckCLError(err, "clEnqueueReadBuffer (GaussianResult)");

    clFinish(m_commandQueue);

    clReleaseMemObject(inputOutputBuffer);
    clReleaseMemObject(tempBuffer);
    clReleaseMemObject(kernelCLBuffer);
}

void GaussianFilter::SetEffectRadius(int radius)
{
    m_effectRadius = std::max(0, radius);
}