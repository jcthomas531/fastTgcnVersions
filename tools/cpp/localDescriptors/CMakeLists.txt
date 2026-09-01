cmake_minimum_required(VERSION 3.10)

project(ld)

find_package(PCL REQUIRED COMPONENTS common io features search)

add_executable(localDescriptors localDescriptors.cpp)

target_link_libraries(localDescriptors ${PCL_LIBRARIES})
