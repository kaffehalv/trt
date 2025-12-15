cmake_minimum_required(VERSION 3.14)
project(trt LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------
# CUDA (for cudart / runtime)
# -----------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------
# TensorRT
# -----------------------------
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h NvOnnxParser.h
    HINTS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
        /usr/local/TensorRT/include
        /usr/local/tensorrt-cuda12.2-8.6.1/include
        /usr/local/tensorrt-cuda12.2-10.4.0.26/include
)

find_library(TENSORRT_LIB_NVINFER
    NAMES nvinfer
    HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/TensorRT/lib
        /usr/local/tensorrt-cuda12.2-8.6.1/lib
        /usr/local/tensorrt-cuda12.2-10.4.0.26/lib
)

find_library(TENSORRT_LIB_NVINFER_PLUGIN
    NAMES nvinfer_plugin
    HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/TensorRT/lib
        /usr/local/tensorrt-cuda12.2-8.6.1/lib
        /usr/local/tensorrt-cuda12.2-10.4.0.26/lib
)

find_library(TENSORRT_LIB_NVONNXPARSER
    NAMES nvonnxparser
    HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/TensorRT/lib
        /usr/local/tensorrt-cuda12.2-8.6.1/lib
        /usr/local/tensorrt-cuda12.2-10.4.0.26/lib
)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_LIB_NVINFER}")
message(STATUS "TensorR nvinfer plugin: ${TENSORRT_LIB_NVINFER_PLUGIN}")
message(STATUS "TensorRT nvonnxparser: ${TENSORRT_LIB_NVONNXPARSER}")

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB_NVINFER OR NOT TENSORRT_LIB_NVONNXPARSER)
    message(FATAL_ERROR "Could not find TensorRT; please adjust paths in CMakeLists.txt")
endif()


# -----------------------------
# Target
# -----------------------------
add_executable(find_best_batch
    find_best_batch.cpp
)

target_include_directories(find_best_batch
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(find_best_batch
    PRIVATE
        ${TENSORRT_LIB_NVINFER}
        ${TENSORRT_LIB_NVINFER_PLUGIN}
        ${TENSORRT_LIB_NVONNXPARSER}
        CUDA::cudart
)
