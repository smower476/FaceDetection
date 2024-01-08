cmake_minimum_required(VERSION 3.13)
project(FaceDetection)

set(CMAKE_CXX_STANDARD 17)

if (UNIX)
	find_package(OpenCV REQUIRED) 
	include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
	add_executable(main main.cpp)
	target_link_libraries(main ${OpenCV_LIBS})
	configure_file(haarcascades/haarcascade_frontalface_default.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endif (UNIX)

if (WIN32)
    find_package(OpenCV REQUIRED) 
	include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
	add_executable(main main.cpp)
	target_link_libraries(main ${OpenCV_LIBS})
	configure_file(haarcascades/haarcascade_frontalface_default.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

endif (WIN32)
