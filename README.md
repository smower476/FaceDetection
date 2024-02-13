## Overview

FaceDetection is an application for detecting faces and drawing an image on their cheeks. Just like masks in social networks!

## Demo

https://github.com/smower476/FaceDetection/assets/121474491/a42c8f76-ee90-4ae8-856e-1c62b34c0b21

## Prerequisites

### CMake

Download and install the [CMake](https://cmake.org/download/) 3.18 or above for your corresponding platform.

### CUDA Toolkit
Download and install the [CUDA Toolkit 12.3](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### OpenCV
Download and build [OpenCV](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html) with CUDA.

## Installing

```
git clone https://github.com/smower476/FaceDetection
cd FaceDetection
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
Or you can build it using Visual Studio and CMake.
## How to use:
1. Copy a videp file and mask (png file) in the same directory with main.exe.
2. Drag&Drop video file to main.exe.
3. Enter mask name (if you want to use it).

## Contributing
Contributions are always welcome! To contribute potential features or bug-fixes:

1. Fork this repository
2. Apply any changes and/or additions based off an existing issue (or create a new issue for the feature/fix you are working on)
3. Create a pull request to have your changes reviewed and merged
