#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <stdexcept> // Для std::runtime_error

// Вспомогательная функция для проверки ошибок OpenCL
void CheckCLError(cl_int errCode, const std::string& operation);

// Вспомогательная функция для загрузки и сборки программы OpenCL
cl_program CreateProgramWithSource(cl_context context, cl_device_id device, const std::string& kernelSource);
