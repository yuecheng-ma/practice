cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(practice)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/3rdparty")

# libtorch
set(LIBTORCH_VERSION "2.6.0")
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/libtorch"
    FIND_PACKAGE_ARGS NAMES Torch REQUIRED PATHS "${FETCHCONTENT_BASE_DIR}/libtorch"
)
FetchContent_MakeAvailable(libtorch)


# add_subdirectory(CUDA)
add_subdirectory(deeplearning)