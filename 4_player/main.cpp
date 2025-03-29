#include "MelodyPlayer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Ошибка: Укажите путь к файлу с мелодией." << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file)
    {
        std::cerr << "Ошибка: Не удалось открыть файл." << std::endl;
        return 1;
    }

    int tempo;
    file >> tempo;

    std::vector<std::string> melody;
    std::string line;

    while (std::getline(file, line))
    {
        if (line == "END")
            break;
        if (!line.empty())
            melody.push_back(line);
    }

    try
    {
        MelodyPlayer player(melody, tempo);
        player.Play();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
    return 0;
}
