cmake_minimum_required(VERSION 3.17)
project(myNN)

set(CMAKE_CXX_STANDARD 14)

include_directories(nn)

add_executable(myNN
        nn/nn.h
        nn/nn_functions.cpp
        nn/nn_grad.cpp
        nn/nn_matrix.cpp
        nn/nn_module.cpp
        nn/nn_tensor.cpp
        nn/nn_var.cpp
        sample.cpp)
