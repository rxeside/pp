cmake_minimum_required(VERSION 3.10)
project(8_3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Укажите пути к вашим установленным OpenCL Headers и ICD Loader, если они не стандартные
list(APPEND CMAKE_PREFIX_PATH "C:/OpenCL/opencl-headers-install") # Пример
list(APPEND CMAKE_PREFIX_PATH "C:/Program Files (x86)/OpenCL-ICD-Loader") # Пример

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED) # Потоки могут быть не нужны, если нет параллелизма CPU

add_executable(8_3
        main.cpp
        MatrixMultiplier.cpp
        GaussianFilter.cpp
        MedianFilter.cpp      # ДОБАВЛЕНО
        MotionBlurFilter.cpp  # ДОБАВЛЕНО
        RadialBlurFilter.cpp
        OpenCLUtils.cpp # Вспомогательные функции для OpenCL
)

target_include_directories(8_3 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${OpenCL_INCLUDE_DIRS} # Используем переменную из find_package(OpenCL)
)

target_link_libraries(8_3 PRIVATE
        OpenCL::OpenCL
        Threads::Threads # Если используется
)

#if (MSVC)
#    target_compile_options(8_3 PRIVATE /W4 /WX)
#else ()
#    target_compile_options(8_3 PRIVATE -Wall -Wextra -Wpedantic -Werror)
#endif ()