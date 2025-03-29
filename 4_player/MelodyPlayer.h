#pragma once

#include <SFML/Audio.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <iostream>
#include <regex>

class MelodyPlayer
{
public:
    MelodyPlayer(const std::vector<std::string>& melody, int bpm);
    void Play();

private:
    std::vector<std::string> m_melody;
    int m_tempo;
    std::unordered_map<std::string, float> m_frequencies;
    sf::SoundBuffer m_buffer;
    sf::Sound m_sound;

    void GenerateWaveform(float frequency, float duration);
    float NoteToFrequency(const std::string& note);
    bool IsValidNote(const std::string& note);
};
