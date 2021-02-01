cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 14)
project(voice_activated_camera VERSION 1.0)

find_package(ALSA REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/test)
