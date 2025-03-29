#include "MelodyPlayer.h"
#include <cmath>
#include <sstream>

MelodyPlayer::MelodyPlayer(const std::vector<std::string>& melody, int bpm)
        : m_melody(melody), m_tempo(bpm)
{
    // карта частот нот
    m_frequencies = {
            {"C0", 16.35f},  {"C#0", 17.32f},  {"D0", 18.35f},  {"D#0", 19.45f},  {"E0", 20.60f},
            {"F0", 21.83f},  {"F#0", 23.12f},  {"G0", 24.50f},  {"G#0", 25.96f},  {"A0", 27.50f},
            {"A#0", 29.14f}, {"B0", 30.87f},   {"C1", 32.70f},  {"C#1", 34.65f},  {"D1", 36.71f},
            {"D#1", 38.89f}, {"E1", 41.20f},   {"F1", 43.65f},  {"F#1", 46.25f},  {"G1", 49.00f},
            {"G#1", 51.91f}, {"A1", 55.00f},   {"A#1", 58.27f}, {"B1", 61.74f},   {"C2", 65.41f},
            {"C#2", 69.30f}, {"D2", 73.42f},   {"D#2", 77.78f}, {"E2", 82.41f},   {"F2", 87.31f},
            {"F#2", 92.50f}, {"G2", 98.00f},   {"G#2", 103.83f},{"A2", 110.00f},  {"A#2", 116.54f},
            {"B2", 123.47f}, {"C3", 130.81f},  {"C#3", 138.59f},{"D3", 146.83f},  {"D#3", 155.56f},
            {"E3", 164.81f}, {"F3", 174.61f},  {"F#3", 185.00f},{"G3", 196.00f},  {"G#3", 207.65f},
            {"A3", 220.00f}, {"A#3", 233.08f}, {"B3", 246.94f}, {"C4", 261.63f},  {"C#4", 277.18f},
            {"D4", 293.66f}, {"D#4", 311.13f}, {"E4", 329.63f}, {"F4", 349.23f},  {"F#4", 369.99f},
            {"G4", 392.00f}, {"G#4", 415.30f}, {"A4", 440.00f}, {"A#4", 466.16f}, {"B4", 493.88f},
            {"C5", 523.25f}
    };
}

void MelodyPlayer::Play()
{
    for (const auto& line : m_melody)
    {
        if (line == "-")
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(60000 / m_tempo));
            continue;
        }

        std::vector<float> frequencies;
        std::stringstream ss(line);
        std::string note;
        while (std::getline(ss, note, '|'))
        {
            if (!IsValidNote(note))
            {
                std::cerr << "Ошибка: Некорректная нота \"" << note << "\". Пропуск." << std::endl;
                continue;
            }
            frequencies.push_back(NoteToFrequency(note));
        }

        GenerateWaveform(frequencies, 60.0f / m_tempo);
        m_sound.play();
        std::this_thread::sleep_for(std::chrono::milliseconds(60000 / m_tempo));
    }
}

void MelodyPlayer::GenerateWaveform(const std::vector<float>& frequencies, float duration)
{
    constexpr int sampleRate = 44100;
    int sampleCount = static_cast<int>(sampleRate * duration);
    std::vector<sf::Int16> samples(sampleCount);

    for (int i = 0; i < sampleCount; ++i)
    {
        float amplitude = 0;
        for (float freq : frequencies)
        {
            amplitude += 30000 * sin(2 * 3.14159f * freq * i / sampleRate);
        }
        amplitude /= frequencies.size();
        samples[i] = static_cast<sf::Int16>(amplitude);
    }

    m_buffer.loadFromSamples(samples.data(), sampleCount, 1, sampleRate);
    m_sound.setBuffer(m_buffer);
}

float MelodyPlayer::NoteToFrequency(const std::string& note)
{
    auto it = m_frequencies.find(note);
    return (it != m_frequencies.end()) ? it->second : 0.0f;
}

bool MelodyPlayer::IsValidNote(const std::string& note)
{
    static std::regex noteRegex(R"([A-G]#?[0-8]-?)");
    return std::regex_match(note, noteRegex);
}
