cmake_minimum_required(VERSION 3.18)
project(FaceDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

file(GLOB HAARCASCADES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/haarcascades/*")
file(COPY ${HAARCASCADES_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/haarcascades)
