cmake_minimum_required(VERSION 3.13)
project(hello)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED) 

include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
