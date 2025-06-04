#pragma once

#include <cstdint>

enum class PacketType : uint8_t
{
    VIDEO_DATA = 0,
};

struct PacketHeader
{
    PacketType type;
    uint64_t timestamp;
    uint32_t videoDataSize;
};