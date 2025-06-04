#pragma once
#include <vector>
#include <string>
#include <CL/cl.h> // Используем C API

class IImageFilter
{
public:
    virtual ~IImageFilter() = default;
    // Применяет фильтр к imageData. imageData может быть изменена по месту.
    virtual void ApplyFilter(
            std::vector<unsigned char>& imageData,
            int width,
            int height,
            int channels) = 0;

    virtual void SetEffectRadius(int radius) = 0;
    virtual std::string GetName() const = 0;

protected:
    // Общие ресурсы OpenCL для фильтров (можно инициализировать в базовом классе или в каждом наследнике)
    // Для простоты, каждый фильтр будет управлять своими ресурсами
};