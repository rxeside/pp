#include "OpenCLUtils.h"
#include <iostream>
#include <vector>
#include <fstream> // Для чтения файла, если бы оно было нужно

void CheckCLError(cl_int errCode, const std::string& operation)
{
    if (errCode != CL_SUCCESS)
    {
        std::cerr << "OpenCL Error during " << operation << ": " << errCode << std::endl;
        // Можно добавить более детальное описание кодов ошибок
        throw std::runtime_error("OpenCL Error: " + operation + " failed with code " + std::to_string(errCode));
    }
}

cl_program CreateProgramWithSource(cl_context context, cl_device_id device, const std::string& kernelSource)
{
    cl_int err;
    const char* sourceStr = kernelSource.c_str();
    size_t sourceSize = kernelSource.length();

    cl_program program = clCreateProgramWithSource(context, 1, &sourceStr, &sourceSize, &err);
    CheckCLError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "--- OpenCL Program Build Log ---" << std::endl;
        std::cerr << buildLog.data() << std::endl;
        std::cerr << "------------------------------" << std::endl;
        CheckCLError(err, "clBuildProgram"); // Это вызовет исключение
    }
    return program;
}
