#pragma once
#include <vector>
#include <string>
#include <CL/cl.h> // C API

class MatrixMultiplier
{
public:
    MatrixMultiplier();
    ~MatrixMultiplier(); // Для освобождения ресурсов OpenCL
    void RunBenchmark(int numRows1, int numColumns1, int numColumns2);

private:
    std::vector<float> MultiplyOnCpu(
            int numRows1, int numColumns1, int numColumns2,
            const std::vector<float>& matrix1, const std::vector<float>& matrix2);

    std::vector<float> MultiplyOnGpu(
            int numRows1, int numColumns1, int numColumns2,
            const std::vector<float>& matrix1, const std::vector<float>& matrix2);

    void InitializeOpenCl();
    void ReleaseOpenCl();
    void PrintMatrixSample(const std::vector<float>& matrix, const std::string& name); // Новая версия

    cl_device_id m_deviceId = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_commandQueue = nullptr;
    cl_program m_program = nullptr;
    cl_kernel m_kernel = nullptr;

    static const int m_tileSize = 16;
    static const std::string m_kernelSource;
};