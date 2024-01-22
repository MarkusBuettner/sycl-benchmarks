cmake_minimum_required(VERSION 3.25)
project(sycl_benchmarks)

set(CMAKE_CXX_STANDARD 17)

find_package(AdaptiveCpp REQUIRED)

add_executable(numa_alloc numa_allocation.cpp)
add_sycl_to_target(TARGET numa_alloc)
