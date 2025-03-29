cmake_minimum_required(VERSION 3.10)
project(MusicPlayer)

add_executable(music_player
        MelodyPlayer.cpp
        main.cpp
)

find_package(SFML 2.5 REQUIRED COMPONENTS audio system)

target_link_libraries(music_player sfml-audio sfml-system)
