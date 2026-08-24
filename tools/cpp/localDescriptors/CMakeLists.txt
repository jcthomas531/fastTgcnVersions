cmake_minimum_required(VERSION 3.10)

project(normals)

find_package(PCL REQUIRED COMPONENTS common io features search)

add_executable(surfNormTest surfNormTest.cpp)

target_link_libraries(surfNormTest ${PCL_LIBRARIES})
