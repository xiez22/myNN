cmake_minimum_required(VERSION 3.5)
project(myNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(myNN
        nn/nn_functions.cpp
        nn/nn_grad.cpp
        nn/nn_matrix.cpp
        nn/nn_module.cpp
        nn/nn_tensor.cpp
        nn/nn_var.cpp
        sample.cpp)

target_include_directories(myNN
        PRIVATE
            nn)
