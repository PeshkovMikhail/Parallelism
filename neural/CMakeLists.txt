cmake_minimum_required(VERSION 3.25)
project(NeuralNetworkCuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit)
find_package(Torch REQUIRED)
add_executable(NeuralNetworkCuda main.cu)

set_target_properties(NeuralNetworkCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(NeuralNetworkCuda CUDA::cublas)
target_link_libraries(NeuralNetworkCuda "${TORCH_LIBRARIES}")
