#include "MatrixMultiplier.h"
#include "GaussianFilter.h"
#include "MedianFilter.h"      // ДОБАВЛЕНО
#include "MotionBlurFilter.h"  // ДОБАВЛЕНО
#include "RadialBlurFilter.h"  // ДОБАВЛЕНО
#include "OpenCLUtils.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


enum class OperationMode
{
    MATRIX_MULTIPLY,
    IMAGE_FILTER
};

struct AppArguments
{
    OperationMode opMode;
    int matrixRows1 = 0;
    int matrixCols1 = 0;
    int matrixCols2 = 0;
    std::string filterTypeName;
    std::string inputImagePath;
    std::string outputImagePath;
    int filterRadius = 5; // Общее название, для motion blur это длина, для radial - интенсивность
};

AppArguments ParseAppArguments(int argc, char* argv[])
{
    AppArguments args;
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " matrix <rows1> <cols1> <cols2>\n"
                  << "  " << argv[0] << " filter <filter_type> <input_image_path> <output_image_path> [parameter_value]\n"
                  << "Filter types: gaussian, median, motion, radial\n"
                  << "Default filter parameter value if not specified: 5\n";
        throw std::runtime_error("Insufficient arguments.");
    }

    std::string modeStr = argv[1];
    if (modeStr == "matrix") {
        args.opMode = OperationMode::MATRIX_MULTIPLY;
        if (argc != 5) throw std::runtime_error("Matrix mode needs 3 dimensions.");
        args.matrixRows1 = std::stoi(argv[2]);
        args.matrixCols1 = std::stoi(argv[3]);
        args.matrixCols2 = std::stoi(argv[4]);
        if (args.matrixRows1 < 1 || args.matrixCols1 < 1 || args.matrixCols2 < 1) {
            throw std::runtime_error("Matrix dimensions must be positive.");
        }
    } else if (modeStr == "filter") {
        args.opMode = OperationMode::IMAGE_FILTER;
        if (argc < 5) throw std::runtime_error("Filter mode needs: filter_type input_path output_path [parameter_value].");
        args.filterTypeName = argv[2];
        args.inputImagePath = argv[3];
        args.outputImagePath = argv[4];
        if (argc > 5) args.filterRadius = std::stoi(argv[5]); // filterRadius - это общее имя для параметра фильтра

        if (args.filterRadius < 0) {
            throw std::runtime_error("Filter parameter (radius/length/intensity) must be non-negative.");
        }
    } else {
        throw std::runtime_error("Unknown mode: " + modeStr);
    }
    return args;
}


int main(int argc, char* argv[])
{
    try
    {
        AppArguments appArgs = ParseAppArguments(argc, argv);

        if (appArgs.opMode == OperationMode::MATRIX_MULTIPLY)
        {
            MatrixMultiplier multiplier;
            multiplier.RunBenchmark(appArgs.matrixRows1, appArgs.matrixCols1, appArgs.matrixCols2);
        }
        else if (appArgs.opMode == OperationMode::IMAGE_FILTER)
        {
            std::cout << "Applying filter: " << appArgs.filterTypeName
                      << " with parameter value " << appArgs.filterRadius // Используем общее имя параметра
                      << "\nInput image: " << appArgs.inputImagePath
                      << "\nOutput image: " << appArgs.outputImagePath
                      << std::endl;

            int width, height, channelsInFile;
            int desiredChannels = 0; // По умолчанию используем то, что в файле

            if (appArgs.filterTypeName == "gaussian") {
                // GaussianFilter ожидает 4 канала (RGBA) из-за uchar4 в ядре
                desiredChannels = 4;
                std::cout << "Note: Gaussian filter will process image as RGBA (4 channels)." << std::endl;
            }
            // Для других фильтров (median, motion, radial) ядра написаны так,
            // чтобы работать с `channels` из файла, поэтому `desiredChannels = 0` (по умолчанию).

            unsigned char *loadedPixels = stbi_load(appArgs.inputImagePath.c_str(), &width, &height, &channelsInFile, desiredChannels);
            if (!loadedPixels) {
                throw std::runtime_error("Failed to load image: " + appArgs.inputImagePath + ". Reason: " + stbi_failure_reason());
            }

            int channelsForProcessing = (desiredChannels == 0) ? channelsInFile : desiredChannels;

            std::cout << "Image loaded: " << width << "x" << height << ", channels in file: " << channelsInFile
                      << ", channels for processing: " << channelsForProcessing << std::endl;

            std::vector<unsigned char> imageData(loadedPixels, loadedPixels + static_cast<size_t>(width) * height * channelsForProcessing);
            stbi_image_free(loadedPixels);

            std::unique_ptr<IImageFilter> imageFilter;

            if (appArgs.filterTypeName == "gaussian") {
                if (channelsForProcessing != 4) {
                    throw std::runtime_error("Gaussian filter internal error: expected 4 channels for processing but got " + std::to_string(channelsForProcessing));
                }
                imageFilter = std::make_unique<GaussianFilter>(appArgs.filterRadius);
            } else if (appArgs.filterTypeName == "median") {
                imageFilter = std::make_unique<MedianFilter>(appArgs.filterRadius);
            } else if (appArgs.filterTypeName == "motion") {
                imageFilter = std::make_unique<MotionBlurFilter>(appArgs.filterRadius);
            } else if (appArgs.filterTypeName == "radial") {
                imageFilter = std::make_unique<RadialBlurFilter>(appArgs.filterRadius);
            }
            else {
                throw std::runtime_error("Unsupported filter type: " + appArgs.filterTypeName);
            }

            imageFilter->ApplyFilter(imageData, width, height, channelsForProcessing);

            std::cout << "Filter '" << imageFilter->GetName() << "' applied." << std::endl;

            int success = 0;
            std::string ext;
            size_t dotPos = appArgs.outputImagePath.rfind('.');
            if (dotPos != std::string::npos) {
                ext = appArgs.outputImagePath.substr(dotPos);
                for(char &c : ext) c = static_cast<char>(tolower(c)); // в нижний регистр
            }

            if (ext == ".png") {
                success = stbi_write_png(appArgs.outputImagePath.c_str(), width, height, channelsForProcessing, imageData.data(), width * channelsForProcessing);
            } else if (ext == ".jpg" || ext == ".jpeg") {
                success = stbi_write_jpg(appArgs.outputImagePath.c_str(), width, height, channelsForProcessing, imageData.data(), 90);
            } else if (ext == ".bmp") {
                success = stbi_write_bmp(appArgs.outputImagePath.c_str(), width, height, channelsForProcessing, imageData.data());
            } else {
                std::cerr << "Warning: Unsupported output file extension '" << ext << "'. Attempting to save as PNG to " << appArgs.outputImagePath << ".png" << std::endl;
                std::string fallbackPath = (dotPos != std::string::npos ? appArgs.outputImagePath.substr(0, dotPos) : appArgs.outputImagePath) + ".png";
                success = stbi_write_png(fallbackPath.c_str(), width, height, channelsForProcessing, imageData.data(), width * channelsForProcessing);
                if (success) appArgs.outputImagePath = fallbackPath;
            }

            if (!success) {
                throw std::runtime_error("Failed to save image to: " + appArgs.outputImagePath);
            }
            std::cout << "Filtered image saved to: " << appArgs.outputImagePath << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "An unknown error occurred." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}