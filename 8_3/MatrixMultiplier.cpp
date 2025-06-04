#include "MatrixMultiplier.h"
#include "OpenCLUtils.h" // Для CheckCLError
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

const std::string MatrixMultiplier::m_kernelSource = R"CLC(
#define TILE_SIZE 16

__kernel void MultiplyMatricesTiled(
    const int numRows1, const int numColumns1, const int numColumns2,
    __global const float* matrix1,
    __global const float* matrix2,
    __global float* resultMatrix) {

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const int numRows2 = numColumns1;

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    const int numTiles = (numColumns1 + TILE_SIZE - 1) / TILE_SIZE;

    float accumulator = 0.0f;
    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx)
    {
        const int tiledARow = globalRow;
        const int tiledACol = tileIdx * TILE_SIZE + localCol;
        if (tiledARow < numRows1 && tiledACol < numColumns1) {
            tileA[localRow][localCol] = matrix1[tiledARow * numColumns1 + tiledACol];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }

        const int tiledBRow = tileIdx * TILE_SIZE + localRow;
        const int tiledBCol = globalCol;
         if (tiledBRow < numRows2 && tiledBCol < numColumns2) {
            tileB[localRow][localCol] = matrix2[tiledBRow * numColumns2 + tiledBCol];
        } else {
            tileB[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            accumulator += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < numRows1 && globalCol < numColumns2) {
        resultMatrix[globalRow * numColumns2 + globalCol] = accumulator;
    }
}
)CLC";

MatrixMultiplier::MatrixMultiplier()
{
    InitializeOpenCl();
    m_program = CreateProgramWithSource(m_context, m_deviceId, m_kernelSource);
    cl_int err;
    m_kernel = clCreateKernel(m_program, "MultiplyMatricesTiled", &err);
    CheckCLError(err, "clCreateKernel (MultiplyMatricesTiled)");
}

MatrixMultiplier::~MatrixMultiplier()
{
    ReleaseOpenCl();
}

void MatrixMultiplier::InitializeOpenCl()
{
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    CheckCLError(err, "clGetPlatformIDs (count)");
    if (numPlatforms == 0) throw std::runtime_error("No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (list)");

    // Просто берем первую платформу и первое GPU устройство на ней
    cl_platform_id platform = platforms[0]; // Упрощение, может потребоваться выбор

    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err == CL_DEVICE_NOT_FOUND) { // Попробуем CPU, если GPU нет
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND) throw std::runtime_error("No GPU or CPU OpenCL devices found.");
        CheckCLError(err, "clGetDeviceIDs (CPU count)");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &m_deviceId, nullptr);
        CheckCLError(err, "clGetDeviceIDs (CPU device)");
        std::cout << "Using OpenCL CPU device." << std::endl;
    } else {
        CheckCLError(err, "clGetDeviceIDs (GPU count)");
        if (numDevices == 0) throw std::runtime_error("No GPU devices found on selected platform.");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_deviceId, nullptr);
        CheckCLError(err, "clGetDeviceIDs (GPU device)");
        std::cout << "Using OpenCL GPU device." << std::endl;
    }


    // Выведем имя устройства для информации
    char deviceName[128];
    clGetDeviceInfo(m_deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Selected device: " << deviceName << std::endl;

    m_context = clCreateContext(nullptr, 1, &m_deviceId, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext");

    // Используем clCreateCommandQueueWithProperties если доступно, иначе старый clCreateCommandQueue
#if defined(CL_VERSION_2_0)
    m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_deviceId, nullptr, &err);
#else
    m_commandQueue = clCreateCommandQueue(m_context, m_deviceId, 0, &err);
#endif
    CheckCLError(err, "clCreateCommandQueue");
}

void MatrixMultiplier::ReleaseOpenCl()
{
    if (m_kernel) clReleaseKernel(m_kernel);
    if (m_program) clReleaseProgram(m_program);
    if (m_commandQueue) clReleaseCommandQueue(m_commandQueue);
    if (m_context) clReleaseContext(m_context);
    // clReleaseDevice не нужен для m_deviceId, он получен, а не создан
}

void MatrixMultiplier::RunBenchmark(int numRows1, int numColumns1, int numColumns2)
{
    std::cout << "Matrix dimensions: A(" << numRows1 << "x" << numColumns1
              << "), B(" << numColumns1 << "x" << numColumns2 << ")" << std::endl;

    std::vector<float> matrix1(numRows1 * numColumns1);
    std::vector<float> matrix2(numColumns1 * numColumns2);

    for (size_t i = 0; i < matrix1.size(); ++i) matrix1[i] = static_cast<float>(i % 100) + 0.1f;
    for (size_t i = 0; i < matrix2.size(); ++i) matrix2[i] = static_cast<float>(i % 50) + 0.2f;

    std::cout << std::fixed << std::setprecision(6);

    auto cpuResult = MultiplyOnCpu(numRows1, numColumns1, numColumns2, matrix1, matrix2);
    PrintMatrixSample(cpuResult, "CPU Result Sample"); // НОВЫЙ ПРАВИЛЬНЫЙ ВЫЗОВ

    auto gpuResult = MultiplyOnGpu(numRows1, numColumns1, numColumns2, matrix1, matrix2);
    PrintMatrixSample(gpuResult, "GPU Result Sample"); // НОВЫЙ ПРАВИЛЬНЫЙ ВЫЗОВ

    bool verified = true;
    if (cpuResult.empty() || gpuResult.empty() || cpuResult.size() != gpuResult.size()) {
        verified = false;
    } else {
        const float epsilon = 1e-3f;
        if (std::abs(cpuResult.front() - gpuResult.front()) > epsilon ||
            std::abs(cpuResult.back() - gpuResult.back()) > epsilon) {
            verified = false;
        }
    }
    std::cout << "Verification: " << (verified ? "PASSED" : "FAILED") << std::endl;
}

std::vector<float> MatrixMultiplier::MultiplyOnCpu(
        int numRows1, int numColumns1, int numColumns2,
        const std::vector<float>& matrix1, const std::vector<float>& matrix2)
{
    std::vector<float> resultMatrix(numRows1 * numColumns2, 0.0f);

    auto startTime = Clock::now();
    for (int i = 0; i < numRows1; ++i)
    {
        for (int j = 0; j < numColumns2; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < numColumns1; ++k)
            {
                sum += matrix1[i * numColumns1 + k] * matrix2[k * numColumns2 + j];
            }
            resultMatrix[i * numColumns2 + j] = sum;
        }
    }
    auto endTime = Clock::now();
    std::cout << "CPU multiplication time: " << Seconds(endTime - startTime).count() << " seconds" << std::endl;
    return resultMatrix;
}

std::vector<float> MatrixMultiplier::MultiplyOnGpu(
        int numRows1, int numColumns1, int numColumns2,
        const std::vector<float>& matrix1, const std::vector<float>& matrix2)
{
    cl_int err;
    std::vector<float> resultMatrix(numRows1 * numColumns2, 0.0f);

    auto startTime = Clock::now();

    cl_mem bufferA = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * matrix1.size(), (void*)matrix1.data(), &err);
    CheckCLError(err, "clCreateBuffer (bufferA)");
    cl_mem bufferB = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * matrix2.size(), (void*)matrix2.data(), &err);
    CheckCLError(err, "clCreateBuffer (bufferB)");
    cl_mem bufferResult = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY,
                                         sizeof(float) * resultMatrix.size(), nullptr, &err);
    CheckCLError(err, "clCreateBuffer (bufferResult)");

    err = clSetKernelArg(m_kernel, 0, sizeof(int), &numRows1); CheckCLError(err, "clSetKernelArg 0");
    err = clSetKernelArg(m_kernel, 1, sizeof(int), &numColumns1); CheckCLError(err, "clSetKernelArg 1");
    err = clSetKernelArg(m_kernel, 2, sizeof(int), &numColumns2); CheckCLError(err, "clSetKernelArg 2");
    err = clSetKernelArg(m_kernel, 3, sizeof(cl_mem), &bufferA); CheckCLError(err, "clSetKernelArg 3");
    err = clSetKernelArg(m_kernel, 4, sizeof(cl_mem), &bufferB); CheckCLError(err, "clSetKernelArg 4");
    err = clSetKernelArg(m_kernel, 5, sizeof(cl_mem), &bufferResult); CheckCLError(err, "clSetKernelArg 5");

    size_t globalWorkSize[2] = {
            static_cast<size_t>((numRows1 + m_tileSize - 1) / m_tileSize * m_tileSize),
            static_cast<size_t>((numColumns2 + m_tileSize - 1) / m_tileSize * m_tileSize)
    };
    size_t localWorkSize[2] = {static_cast<size_t>(m_tileSize), static_cast<size_t>(m_tileSize)};

    err = clEnqueueNDRangeKernel(m_commandQueue, m_kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    CheckCLError(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(m_commandQueue, bufferResult, CL_TRUE, 0,
                              sizeof(float) * resultMatrix.size(), resultMatrix.data(), 0, nullptr, nullptr);
    CheckCLError(err, "clEnqueueReadBuffer");

    clFinish(m_commandQueue); // Убедимся, что все выполнено

    auto endTime = Clock::now();
    std::cout << "GPU multiplication time: " << Seconds(endTime - startTime).count() << " seconds" << std::endl;

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);

    return resultMatrix;
}

void MatrixMultiplier::PrintMatrixSample(const std::vector<float>& matrix, const std::string& name) { // Новая версия
    if (matrix.empty()) {
        std::cout << name << " is empty." << std::endl;
        return;
    }
    std::cout << name << " (first element): " << matrix[0] << std::endl;
    if (matrix.size() > 1) {
        std::cout << name << " (last element): " << matrix.back() << std::endl;
    }
    // Если захотите выводить часть матрицы, вам понадобятся numRows и numCols:
    // std::cout << name << " (sample " << std::min(numRows, 3) << "x" << std::min(numCols, 3) << "):\n";
    // for (int i = 0; i < std::min(numRows, 3); ++i) {
    //     for (int j = 0; j < std::min(numCols, 3); ++j) {
    //         if (i * numCols + j < matrix.size()) { // Проверка границ
    //             std::cout << std::setw(8) << matrix[i * numCols + j] << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
}