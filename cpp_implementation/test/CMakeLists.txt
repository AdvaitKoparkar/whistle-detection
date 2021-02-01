add_executable (hello_world ${PROJECT_SOURCE_DIR}/test/hello_world.cpp)

include_directories(${ALSA_INCLUDE_DIRECTORIES})
target_link_libraries(hello_world ${ALSA_LIBRARIES})
