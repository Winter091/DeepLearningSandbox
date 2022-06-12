cmake_minimum_required(VERSION 3.10)

project(DeepLearning)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    DeepLearning

    Source/main.cpp
    Source/Pool.cpp
    Source/Model.cpp
    Source/ModelLearn.cpp
    Source/ModelVisualizer.cpp
)
